# Incident Review Packet

This document is a realistic exploratory input for harder `eu_pii` families. It is synthetic and intended for manual inspection rather than automated regression.

## Intake Summary

Laura Gómez Serrano from Atlas Soluciones S.L. opened a remediation review after a support export included customer information in plain text. The ticket named Laura Gómez Serrano as the reporting contact, used the mailbox laura.gomez@atlas-soluciones.es, listed the mobile number +34 612 45 78 90, and repeated the office address Calle de Alcalá 147, 28009 Madrid. The packet also referenced Atlas Soluciones S.L. several times because the escalation crossed support, billing, and privacy operations.

Javier Ortega Ruiz joined the same review as the privacy lead. The notes repeated Javier Ortega Ruiz, the identifier 54839271H, the backup number +34 633 10 44 21, and the same Atlas Soluciones S.L. organization name because the case had to be rechecked by a second team. One paragraph also preserved the public IP 83.56.144.201 and the portal URL https://portal.atlas-soluciones.es/case/4827.

## Customer Timeline

At 08:12, Laura Gómez Serrano confirmed that laura.gomez@atlas-soluciones.es was copied into a free-text field that should have contained only an internal case number. The transcript also repeated +34 612 45 78 90 and Calle de Alcalá 147, 28009 Madrid because the operator pasted the same signature block twice while explaining the leak. Atlas Soluciones S.L. appears again in the copied footer.

At 08:26, Javier Ortega Ruiz reviewed the same thread and restated that 54839271H, +34 633 10 44 21, and 83.56.144.201 should never appear in downstream exports. The reviewer pasted the same portal link https://portal.atlas-soluciones.es/case/4827 and then repeated Laura Gómez Serrano and Atlas Soluciones S.L. while summarizing the impact.

## Compliance Note

The compliance note bundled one payment example and one contact example into the same paragraph. It explicitly repeated Laura Gómez Serrano, laura.gomez@atlas-soluciones.es, +34 612 45 78 90, Calle de Alcalá 147, 28009 Madrid, Atlas Soluciones S.L., and the IBAN ES91 2100 0418 4502 0005 1332 so the reviewers could inspect how the anonymization node behaves on a realistic dossier. The same paragraph also repeated Javier Ortega Ruiz, 54839271H, +34 633 10 44 21, and https://portal.atlas-soluciones.es/case/4827.

## Escalation Mail

From: laura.gomez@atlas-soluciones.es
To: privacy@atlas-soluciones.es
Subject: Export still contains customer records

Laura Gómez Serrano wrote that Atlas Soluciones S.L. still had one export containing Calle de Alcalá 147, 28009 Madrid, +34 612 45 78 90, and ES91 2100 0418 4502 0005 1332 in plain text. Javier Ortega Ruiz replied that 54839271H and 83.56.144.201 were visible in the same package. The reply quoted the same URL https://portal.atlas-soluciones.es/case/4827 again for review.
