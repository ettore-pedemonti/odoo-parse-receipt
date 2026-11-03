"""Receipt extension: Parse photos, the Read labels button and automatic filling of the receipt."""
from markupsafe import Markup

from odoo import fields, models


class StockPicking(models.Model):
    _inherit = "stock.picking"

    parse_photo_ids = fields.One2many("parse.receipt.photo", "picking_id", string="Parse photos")

    def action_parse_upload(self):
        """Open the photo upload dialog on top of the receipt."""
        self.ensure_one()
        return {
            "type": "ir.actions.act_window",
            "name": self.env._("Read labels"),
            "res_model": "parse.receipt.upload",
            "view_mode": "form",
            "target": "new",
            "context": {"default_picking_id": self.id},
        }

    def _parse_read_pending(self):
        """Read the queued photos with Parse, then fill in the receipt."""
        self.ensure_one()
        pending = self.parse_photo_ids.filtered(lambda photo: photo.state in ("to_read", "error"))
        if pending:
            Photo = self.env["parse.receipt.photo"]
            catalog = Photo._parse_catalog()          # built once for all photos
            label_schema = Photo._parse_label_schema()
            read_count = error_count = 0
            for photo in pending:
                # an error on one photo never stops the others: it is stored on the photo
                data, error = Photo._parse_call(photo.attachment_id, catalog, label_schema)
                photo._parse_store(data, error)
                if error:
                    error_count += 1
                else:
                    read_count += 1
            self.message_post(body=self.env._("Parse: %(read)s photos read, %(errors)s with errors.",
                                              read=read_count, errors=error_count))
        self._parse_compile()

    def _parse_compile(self):
        """After reading: supplier, receipt lines, photo warnings and chatter summary."""
        self.ensure_one()
        # photos read but not yet turned into lines: the ones to report in this run
        to_report = self.parse_photo_ids.filtered(lambda photo: photo.state == "done" and not photo.applied)
        self._parse_apply_supplier()
        self._parse_apply_lines()
        self._parse_update_warnings()
        failed = self.parse_photo_ids.filtered(lambda photo: photo.state == "error")
        self._parse_post_warnings(to_report, failed)

    def _parse_post_warnings(self, read_photos, failed_photos):
        """Chatter message listing photos with warnings or errors: what to check before Validate."""
        _ = self.env._
        with_warnings = read_photos.filtered("warning")
        if not with_warnings and not failed_photos:
            return
        # Markup % (...) escapes the values: a file name cannot inject HTML
        items = Markup("").join(
            Markup("<li><b>%s</b>: %s</li>") % (photo.name, photo.warning) for photo in with_warnings
        ) + Markup("").join(
            Markup("<li><b>%s</b>: %s</li>") % (
                photo.name,
                _("reading failed (%s). Use Read labels > Upload and read to retry, or add the line manually",
                  photo.error_message or ""),
            )
            for photo in failed_photos
        )
        title = _("Parse: data to check before Validate")
        self.message_post(body=Markup("<p>⚠️ %s</p><ul>%s</ul>") % (title, items))

    def _parse_apply_supplier(self):
        """Empty Receive From and one recognised supplier: the module fills it in, not the operator."""
        self.ensure_one()
        suppliers = self.parse_photo_ids.filtered(lambda photo: photo.state == "done").supplier_id
        if not self.partner_id and len(suppliers) == 1:
            self.partner_id = suppliers
        if len(suppliers) > 1:
            self.message_post(body=self.env._("Parse: photos from different suppliers on this receipt: %s",
                                              ", ".join(suppliers.mapped("name"))))

    def _parse_apply_lines(self):
        """Create the receipt lines from the read photos: one move per product, one detailed line per lot."""
        self.ensure_one()
        _ = self.env._
        new_photos = self.parse_photo_ids.filtered(lambda photo: photo.state == "done" and not photo.applied)
        new_photos._parse_link_records()
        with_product = new_photos.filtered("product_id")
        if not with_product:
            return

        # 1) group the photos by (product, lot): same lot = quantities added up
        groups = {}
        for photo in with_product:
            values = photo._parse_label_values()
            key = (photo.product_id.id, values["lot"])
            group = groups.setdefault(key, {
                "product": photo.product_id, "lot": values["lot"], "quantity": 0.0,
                "expiries": set(), "photos": self.env["parse.receipt.photo"], "warnings": [],
                "expiry_flagged": False,
            })
            group["photos"] |= photo
            group["quantity"] += values["quantity"] or 0.0
            if values["expiry"]:
                group["expiries"].add(values["expiry"])
            group["warnings"] += values["warnings"]
            group["expiry_flagged"] |= values["expiry_flagged"]

        # 2) one move per product: an existing one (e.g. from a purchase order) is reused
        for group in groups.values():
            product = group["product"]
            move = self.move_ids.filtered(
                lambda m: m.product_id == product and m.state not in ("done", "cancel"))[:1]
            if not move:
                self.env["stock.move"].create({
                    "picking_id": self.id,
                    "product_id": product.id,
                    "product_uom": product.uom_id.id,
                    "product_uom_qty": sum(g["quantity"] for g in groups.values() if g["product"] == product),
                })
        # draft moves -> confirmed, as with "Mark as To Do": lots can be entered from here on
        if self.move_ids.filtered(lambda m: m.state == "draft"):
            self.action_confirm()

        # 3) on confirmation Odoo creates one line without lot per move: removed once, replaced by ours
        products = self.env["product.product"].browse({key[0] for key in groups})
        moves = self.move_ids.filtered(lambda m: m.product_id in products and m.state not in ("done", "cancel"))
        moves.move_line_ids.filtered(lambda line: not line.lot_id and not line.lot_name).unlink()

        # 4) one detailed line per lot, with quantity and expiry date
        MoveLine = self.env["stock.move.line"]
        for group in groups.values():
            product, lot = group["product"], group["lot"]
            move = moves.filtered(lambda m: m.product_id == product)[:1]
            expiry = max(group["expiries"]) if group["expiries"] else False
            existing = move.move_line_ids.filtered(lambda line: lot and line.lot_name == lot)[:1]
            if existing:
                existing.quantity += group["quantity"]
            else:
                values = {
                    "move_id": move.id,
                    "picking_id": self.id,
                    "product_id": product.id,
                    "product_uom_id": product.uom_id.id,
                    "location_id": move.location_id.id,
                    "location_dest_id": move.location_dest_id.id,
                    "quantity": group["quantity"],
                    "lot_name": lot or False,
                }
                if product.use_expiration_date and expiry:
                    values["expiration_date"] = expiry
                MoveLine.create(values)

            warnings = list(group["warnings"])
            if len(group["photos"]) > 1:
                warnings.append(_("%s photos with the same lot: check the number of containers", len(group["photos"])))
            if not lot:
                warnings.append(_("Lot not read: enter it manually"))
            if len(group["expiries"]) > 1:
                warnings.append(_("Different expiry dates on the same lot: check them"))
            # generic warning only when there is no more specific one about the expiry date
            if product.use_expiration_date and not expiry and not group["expiry_flagged"]:
                warnings.append(_("Expiry date not read: enter it manually"))
            group["photos"].write({"applied": True, "line_warning": "; ".join(dict.fromkeys(warnings)) or False})

        self.message_post(body=_("Parse: %(lines)s lines filled in from %(photos)s photos.",
                                 lines=len(groups), photos=len(with_product)))

    def _parse_update_warnings(self):
        """Recompute from scratch the warnings of every read photo."""
        self.ensure_one()
        for photo in self.parse_photo_ids.filtered(lambda photo: photo.state == "done"):
            warnings = photo._parse_base_warnings()
            # supplier already set (e.g. from a purchase order) and different: not overwritten, flagged
            if photo.supplier_id and self.partner_id and photo.supplier_id != self.partner_id:
                warnings.append(self.env._("Label: %(label)s; receipt: %(receipt)s",
                                           label=photo.supplier_id.name, receipt=self.partner_id.name))
            if photo.line_warning:
                warnings.append(photo.line_warning)
            photo.warning = "; ".join(warnings) or False
