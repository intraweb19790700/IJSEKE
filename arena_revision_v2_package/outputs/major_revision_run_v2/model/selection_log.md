# Lambda Selection Log (validation only)

Rule (preregistered in `model/selection_config.yaml` before training): maximize validation composite macro-F1; within tolerance 0.005 maximize validation mean consistency; then choose smaller lambda. Locked-test labels/predictions were never used.

 lambda  val_composite_macro_f1_mean  val_composite_macro_f1_std  val_mean_consistency_mean  val_mean_consistency_std  seeds
   0.00                     0.549485                    0.003285                   0.961462                  0.002866      5
   0.03                     0.550260                    0.003081                   0.961462                  0.002130      5
   0.05                     0.550492                    0.003343                   0.961905                  0.003190      5
   0.10                     0.550277                    0.003763                   0.962791                  0.001853      5
   0.25                     0.550396                    0.002864                   0.964341                  0.003947      5

Best validation composite macro-F1: 0.550492
**Selected lambda: 0.25**
