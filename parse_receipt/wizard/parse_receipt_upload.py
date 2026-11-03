"""Photo upload dialog: on an existing receipt (Read labels) or standalone (Receipt from photos)."""
from odoo import fields, models
from odoo.exceptions import UserError


class ParseReceiptUpload(models.TransientModel):
    _name = "parse.receipt.upload"
    _description = "Parse photo upload"

    # empty = Receipt from photos: the module creates the receipts
    picking_id = fields.Many2one("stock.picking", string="Receipt")
    # the many2many_binary widget creates the attachments as soon as files are uploaded
    attachment_ids = fields.Many2many("ir.attachment", string="Photos")

    def action_load(self):
        """Upload and read button."""
        self.ensure_one()
        if self.picking_id:
            return self._load_on_picking()
        return self._load_new_receipts()

    def _split_duplicates(self, seen_checksums):
        """Split the uploaded photos into new ones and duplicates, comparing checksums."""
        new = self.env["ir.attachment"]
        duplicates = self.env["ir.attachment"]
        for attachment in self.attachment_ids:
            if attachment.checksum in seen_checksums:
                duplicates |= attachment
            else:
                seen_checksums.add(attachment.checksum)    # also catches two copies in the same upload
                new |= attachment
        return new, duplicates

    def _load_on_picking(self):
        """Read labels on a receipt: attach the photos, then read them and fill in the receipt."""
        picking = self.picking_id
        new, duplicates = self._split_duplicates(set(picking.parse_photo_ids.mapped("checksum")))
        # the attachment moves from the dialog to the receipt, so it also shows in the chatter
        new.write({"res_model": "stock.picking", "res_id": picking.id})
        self.env["parse.receipt.photo"].create(
            [{"picking_id": picking.id, "attachment_id": attachment.id} for attachment in new]
        )
        if duplicates:
            picking.message_post(body=self.env._("Parse: photos discarded, already uploaded on this receipt: %s",
                                                 ", ".join(duplicates.mapped("name"))))
            duplicates.unlink()
        # new photos and photos left in error by a previous attempt
        picking._parse_read_pending()
        return {"type": "ir.actions.act_window_close"}    # the receipt underneath reloads

    def _load_new_receipts(self):
        """Receipt from photos: read the photos, group them by supplier, create one receipt per group."""
        _ = self.env._
        new, duplicates = self._split_duplicates(set())
        duplicate_names = ", ".join(duplicates.mapped("name"))
        duplicates.unlink()
        if not new:
            raise UserError(_("Upload at least one photo."))

        # 1) read every photo with Parse
        Photo = self.env["parse.receipt.photo"]
        catalog = Photo._parse_catalog()
        label_schema = Photo._parse_label_schema()
        groups = {}                             # supplier id (False if not recognised) -> readings
        for attachment in new:
            data, error = Photo._parse_call(attachment, catalog, label_schema)
            supplier_name = ((data or {}).get("supplier") or {}).get("name")
            partner = self.env["res.partner"]
            if supplier_name:
                partner = partner.search([("name", "=", supplier_name)], limit=1)
            groups.setdefault(partner.id, []).append((attachment, data, error))

        # 2) one receipt per supplier, with the Receipts operation type of the company warehouse
        picking_type = self.env["stock.warehouse"].search(
            [("company_id", "=", self.env.company.id)], limit=1).in_type_id
        if not picking_type:
            raise UserError(_("No Receipts operation type found for this company."))
        receipts = self.env["stock.picking"]
        for partner_id, readings in groups.items():
            receipt = self.env["stock.picking"].create({
                "picking_type_id": picking_type.id,
                "partner_id": partner_id or False,      # supplier not recognised: left empty
            })
            for attachment, data, error in readings:
                attachment.write({"res_model": "stock.picking", "res_id": receipt.id})
                photo = Photo.create({"picking_id": receipt.id, "attachment_id": attachment.id})
                photo._parse_store(data, error)     # already read: store the outcome without calling Parse again
            receipt.message_post(body=_("Parse: receipt created from %s photos (Receipt from photos).", len(readings)))
            receipt._parse_compile()
            receipts |= receipt
        if duplicate_names:
            receipts[:1].message_post(body=_("Parse: photos discarded, uploaded twice: %s", duplicate_names))

        # 3) open the new receipt, or the list when there are several
        if len(receipts) == 1:
            return {
                "type": "ir.actions.act_window",
                "res_model": "stock.picking",
                "res_id": receipts.id,
                "views": [(False, "form")],
                "target": "current",
            }
        return {
            "type": "ir.actions.act_window",
            "name": _("Receipts from photos"),
            "res_model": "stock.picking",
            "domain": [("id", "in", receipts.ids)],
            "views": [(False, "list"), (False, "form")],
            "target": "current",
        }
