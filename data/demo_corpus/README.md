# Demo corpus (synthetic)

These 13 documents are **synthetic**. They describe a fictional freight brokerage,
"Larkspur Freight", created for this demo. They contain no client data, no real
company, no real person and no real address. Any resemblance to a real carrier is
coincidental.

The full 30-question evaluation uses a 57-document private corpus that is not
published. This folder exists so a reviewer who clones the repository can run the
same pipeline end to end without that corpus and without Postgres.

Domain: dry-van and reefer freight brokerage operations. The facts are deliberately
specific (rates, time windows, limits) so the deterministic scorer can check them.
