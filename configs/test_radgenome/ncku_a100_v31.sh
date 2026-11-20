# V3.1 = V3 (GT v3, frozen vitq, same report CSVs) with lesion slots only for the nodules the
# report describes: match_tier A/B/C/X in nodule_metadata_v2.csv. 'none' nodules (a mask no
# sentence was matched to; mostly < 5 mm, RADS 1, left out of the report) get no lesion token.
# In V3, 21% of the train lobes with lesion tokens had a normal target, which taught the model
# to skip lesions it was given. Oracle boxes here = the nodules the radiologist wrote up.
#   bash test_radgenome.sh ncku_a100_v31      # split=val | test_me | test_ldct, ckpt_step=...
experiment_name="${experiment_name:-Reg2RG_v31_vitq}"
lesion_tiers="${lesion_tiers-A,B,C,X}"
source "$(dirname "${BASH_SOURCE[0]}")/ncku_a100_v3.sh"
