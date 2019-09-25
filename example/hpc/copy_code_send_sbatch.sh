#!/bin/bash
#set -x

HPCLOCATION="/home/jtb34/rds/rds-jtb34-gacf/GACF_OUTPUTS/YEET"
XMATCHHPCLOCATION="$HPCLOCATION/cross_match"
NGTSFIELDRUN="/home/jtb34/GitHub/GACF/example/hpc/NGTSFieldRun.py"
SBATCHSCRIPT="/home/jtb34/GitHub/GACF/example/hpc/example_submission.peta4-skylake"

HPCDIR="${HPCLOCATION//YEET/$FIELDNAME}"
HPCXMATCHDIR="${XMATCHHPCLOCATION//YEET/$FIELDNAME}"

if [ "$1" != "" ]; then
    echo "Using fieldname $1"
    FIELDNAME=$1
else
    echo "Enter fieldname (e.g. NG0004-2950):"
    read FIELDNAME
fi

cp "$NGTSFIELDRUN" "$HPCDIR"
sed "s/YEET/$FIELDNAME/g" "$SBATCHSCRIPT" > "$HPCDIR/$FIELDNAME.peta4-skylake"

if [ "$2" != "" ]; then
    case $2 in
        [Yy]* ) echo "Sending sbatch"; sbatch "$HPCDIR/$FIELDNAME.peta4-skylake";;
        [Nn]* ) echo "Not sending sbatch"; echo "sbatch $HPCDIR/$FIELDNAME.peta4-skylake"; exit;;
        * ) echo "Input not recognised, send sbatch manually";;
    esac
else
    while true; do
        read -p "Send sbatch command?" yn
        case $yn in
            [Yy]* ) sbatch "$HPCDIR/$FIELDNAME.peta4-skylake"; break;;
            [Nn]* ) echo "Not sending sbatch"; echo "sbatch $HPCDIR/$FIELDNAME.peta4-skylake"; exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done
fi