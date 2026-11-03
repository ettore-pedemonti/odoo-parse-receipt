"""Parse photo: one container-label photo attached to a receipt, with its reading status."""
import base64
import calendar
import json
import logging
from datetime import datetime, timedelta

import pytz
import requests

from odoo import api, fields, models
from odoo.tools import file_open

_logger = logging.getLogger(__name__)

DEFAULT_SERVICE_URL = "http://parse:8000"    # service name in compose.yaml, internal port
# The service runs two AI steps, each with a 60 s timeout per call and one retry
REQUEST_TIMEOUT_SEC = 300

# Units the label schema allows -> Odoo units (XML ids of the uom module)
LABEL_UNITS = {
    "kg": "uom.product_uom_kgm",
    "g": "uom.product_uom_gram",
    "l": "uom.product_uom_litre",
    "ml": "uom.product_uom_milliliter",
}
# Allowed conversions: weight to weight, volume to volume (litres to grams would need the density)
UNIT_FAMILIES = ({"kg", "g"}, {"l", "ml"})
# Minimum share of agents agreeing on a field: with 3 agents, 2 out of 3 (0.67) passes, 1 out of 3 does not
MIN_AGREEMENT = 0.6
# A plausible expiry date is not in the past and not more than this many years ahead
MAX_EXPIRY_YEARS = 10


class ParseReceiptPhoto(models.Model):
    _name = "parse.receipt.photo"
    _description = "Parse label photo"
    _order = "id"

    picking_id = fields.Many2one("stock.picking", string="Receipt", required=True, ondelete="cascade", index=True)
    attachment_id = fields.Many2one("ir.attachment", string="Photo", required=True, ondelete="cascade")
    name = fields.Char(related="attachment_id.name", string="File name")
    # SHA1 of the file content, computed by Odoo for every attachment: same file = same checksum
    checksum = fields.Char(related="attachment_id.checksum", store=True, index=True)
    state = fields.Selection(
        [("to_read", "To read"), ("done", "Read"), ("error", "Error")],
        string="Status", default="to_read", required=True,
    )
    error_message = fields.Text(string="Error")
    label_text = fields.Char(string="Read on label")
    confidence = fields.Float(string="Confidence")             # share of agents agreeing on the product
    result_json = fields.Text(string="Parse response")         # full response, for troubleshooting
    product_id = fields.Many2one("product.product", string="Product")
    supplier_id = fields.Many2one("res.partner", string="Supplier read")
    applied = fields.Boolean(string="Line created")            # already turned into a receipt line
    line_warning = fields.Char(string="Line warning")          # warnings raised while creating the lines
    warning = fields.Char(string="Warning")                    # all warnings: what the operator must check

    # ------------------------------------------------------------------
    # Call to the Parse service
    # ------------------------------------------------------------------

    @api.model
    def _parse_catalog(self):
        """Catalog sent to Parse: code, name and suppliers of every product with an internal reference."""
        products = self.env["product.product"].search([("default_code", "!=", False)])
        return [
            {
                "code": product.default_code,
                "name": product.name,
                "suppliers": sorted(set(product.product_tmpl_id.seller_ids.partner_id.mapped("name"))),
            }
            for product in products
        ]

    @api.model
    def _parse_label_schema(self):
        """JSON Schema of the fields Parse must read on the label."""
        with file_open("parse_receipt/schemas/label.json") as schema_file:
            return json.load(schema_file)

    @api.model
    def _parse_call(self, attachment, catalog, label_schema):
        """Send one photo to Parse. Returns (response, None) on success, (None, message) on failure."""
        # The address can be changed without code: Settings > Technical > System Parameters
        url = self.env["ir.config_parameter"].sudo().get_param("parse_receipt.service_url", DEFAULT_SERVICE_URL)
        payload = {
            "photo": base64.b64encode(attachment.raw).decode("ascii"),
            "label_schema": label_schema,
            "catalog": catalog,
        }
        try:
            response = requests.post(f"{url}/read", json=payload, timeout=REQUEST_TIMEOUT_SEC)
        except requests.exceptions.ConnectionError:
            return None, self.env._("Parse service unreachable")
        except requests.exceptions.Timeout:
            return None, self.env._("No answer from Parse within %s seconds", REQUEST_TIMEOUT_SEC)
        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text)   # FastAPI puts the reason in "detail"
            except ValueError:
                detail = response.text
            return None, self.env._("Parse error %(status)s: %(detail)s",
                                    status=response.status_code, detail=str(detail)[:500])
        return response.json(), None

    def _parse_store(self, data, error):
        """Store the outcome of the reading on the photo."""
        self.ensure_one()
        if error:
            _logger.warning("Parse: photo %s failed: %s", self.name, error)
            self.write({"state": "error", "error_message": error})
            return
        choice = data.get("choice") or {}
        _logger.info("Parse: photo %s read, product %s", self.name, choice.get("code"))
        self.write({
            "state": "done",
            "error_message": False,
            "label_text": data["reading"].get("printed_name") or "",
            "confidence": choice.get("confidence") or 0.0,
            "result_json": json.dumps(data, ensure_ascii=False, indent=2),
        })
        self._parse_link_records()

    # ------------------------------------------------------------------
    # From the Parse response to Odoo data
    # ------------------------------------------------------------------

    def _parse_data(self):
        """Parse response stored on the photo, as a dict."""
        self.ensure_one()
        return json.loads(self.result_json or "{}")

    def _parse_link_records(self):
        """Turn the code and supplier name returned by Parse into Odoo records."""
        for photo in self:
            data = photo._parse_data()
            code = (data.get("choice") or {}).get("code")
            supplier_name = (data.get("supplier") or {}).get("name")
            product = self.env["product.product"]
            if code:
                product = product.search([("default_code", "=", code)], limit=1)
            partner = self.env["res.partner"]
            if supplier_name:
                partner = partner.search([("name", "=", supplier_name)], limit=1)
            photo.write({"product_id": product.id, "supplier_id": partner.id})

    def _parse_label_values(self):
        """Lot, expiry date and quantity (in the product unit) read on the label, plus warnings."""
        self.ensure_one()
        reading = self._parse_data().get("reading") or {}
        _ = self.env._
        warnings = []

        def agreement(field):
            """Share of agents agreeing on the field (<field>_% in the Parse response); 1.0 if absent."""
            value = reading.get(f"{field}_%")
            return 1.0 if value is None else float(value)

        def readings(field):
            """Different values proposed by the agents, shown in the warning (e.g. 2000, 2024, 2026)."""
            return ", ".join(str(value) for value in (reading.get(f"{field}_opts") or {}))

        # Lot
        lot = (reading.get("lot") or "").strip()
        if lot and agreement("lot") < MIN_AGREEMENT:
            warnings.append(_("Uncertain lot, different readings (%s): enter it manually", readings("lot")))
            lot = ""
        elif lot and agreement("lot") < 1:
            warnings.append(_("Check the lot (%s)", readings("lot")))

        # Expiry date: with month and year only, the last day of the month is used
        expiry = None
        warnings_before_expiry = len(warnings)
        year, month, day = reading.get("expiry_year"), reading.get("expiry_month"), reading.get("expiry_day")
        date_fields = ("expiry_day", "expiry_month", "expiry_year")
        date_agreement = min(agreement(field) for field in date_fields)
        if year and month and date_agreement < MIN_AGREEMENT:
            # agents disagree on the date: no expiry date is better than a wrong one
            proposals = "; ".join(f"{field.split('_')[1]}: {readings(field)}"
                                  for field in date_fields if agreement(field) < MIN_AGREEMENT)
            warnings.append(_("Uncertain expiry date, different readings (%s): enter it manually", proposals))
        elif year and month:
            try:
                day = day or calendar.monthrange(int(year), int(month))[1]
                # Odoo stores datetimes in UTC: take midnight in the user's timezone and convert it,
                # so the lot shows "31 Jan 2027, 00:00" and not "01:00"
                timezone = pytz.timezone(self.env.context.get("tz") or self.env.user.tz or "UTC")
                local = timezone.localize(datetime(int(year), int(month), int(day)))
                expiry = local.astimezone(pytz.utc).replace(tzinfo=None)
            except (ValueError, TypeError, pytz.UnknownTimeZoneError):
                warnings.append(_("Invalid expiry date on the label: %s", f"{day}/{month}/{year}"))
            if expiry:
                now = datetime.now()
                shown = f"{int(day):02d}/{int(month):02d}/{year}"
                if expiry < now:
                    warnings.append(_("Expiry date already past (%s): check the label", shown))
                    expiry = None                  # a past expiry date is never written
                elif expiry > now + timedelta(days=365 * MAX_EXPIRY_YEARS):
                    warnings.append(_("Expiry date more than %(years)s years ahead (%(date)s): check the label",
                                      years=MAX_EXPIRY_YEARS, date=shown))
                    expiry = None
                elif date_agreement < 1:
                    warnings.append(_("Check the expiry date (%s)", shown))

        expiry_flagged = len(warnings) > warnings_before_expiry     # a specific expiry warning was raised

        # Quantity, converted from the label unit to the product unit (5 kg -> 5000 g)
        quantity = None
        value, unit = reading.get("net_quantity"), reading.get("unit")
        product_uom = self.product_id.uom_id
        product_unit = next(
            (key for key, xml_id in LABEL_UNITS.items()
             if self.env.ref(xml_id, raise_if_not_found=False) == product_uom),
            None,
        )
        if value is not None and min(agreement("net_quantity"), agreement("unit")) < MIN_AGREEMENT:
            warnings.append(_("Uncertain quantity, different readings (%s): enter it manually", readings("net_quantity")))
        elif value is None or unit not in LABEL_UNITS:
            warnings.append(_("Quantity not read: enter it manually"))
        elif not any(unit in family and product_unit in family for family in UNIT_FAMILIES):
            warnings.append(_("Label in %(unit)s, product in %(uom)s: enter the quantity manually",
                              unit=unit, uom=product_uom.name))
        else:
            quantity = self.env.ref(LABEL_UNITS[unit])._compute_quantity(float(value), product_uom)
            if min(agreement("net_quantity"), agreement("unit")) < 1:
                warnings.append(_("Check the quantity (%s)", f"{value} {unit}"))

        return {"lot": lot, "expiry": expiry, "quantity": quantity, "warnings": warnings,
                "expiry_flagged": expiry_flagged}

    def _parse_base_warnings(self):
        """Warnings that depend only on the photo: product, agreement of the agents, supplier."""
        self.ensure_one()
        _ = self.env._
        data = self._parse_data()
        choice = data.get("choice") or {}
        supplier = data.get("supplier") or {}
        warnings = []
        if not self.product_id:
            warnings.append(_("Product not recognised: add the line manually"))
        elif (choice.get("confidence") or 0) < 1:
            warnings.append(_("Agents disagree on the product: check it"))
        if not self.supplier_id:
            warnings.append(_("Supplier not recognised (read: %s)", supplier.get("read") or "-"))
        return warnings
