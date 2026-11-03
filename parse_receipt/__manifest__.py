{
    'name': 'Parse - Goods Receipt from Label Photos',
    'version': '19.0.1.0.0',
    'summary': 'Fill in receipts automatically from photos of container labels, read by AI',
    'description': """
Goods receipt from label photos
===============================
The operator photographs the labels of the incoming containers. The module sends each photo to the
Parse service, which reads supplier, product, lot, expiry date and net quantity with several AI agents
and matches the product against the catalog. The receipt is then filled in automatically:
supplier, one move per product, one detailed line per lot with expiry date and quantity converted to
the product unit. Uncertain or missing data is never written: it is flagged in the chatter.
""",
    'category': 'Inventory/Inventory',
    'author': 'Ettore Pedemonti',
    'license': 'LGPL-3',
    'depends': ['stock'],
    'external_dependencies': {'python': ['requests']},
    'data': [
        'security/ir.model.access.csv',
        'wizard/parse_receipt_upload_views.xml',
        'views/stock_picking_views.xml',
    ],
    'installable': True,
    'application': False,
}
