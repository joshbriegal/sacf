#!/bin/bash
#set -x

# Arguments:
# 1: Fieldname
# 2: Submit Sbatch job (y/n)?
# 3: Copy data from server (y/n)?
# 4: Overwrite data (if files already exist) (y/n)?


NGPHOTGETINFO="/home/jtb34/GitHub/GACF/example/hpc/ngtshead_cmds.txt"
NGTSHEAD="ngtshead.warwick.ac.uk"
FITSLOCATION="/home/jtb34/tools"
XMATCHLOCATION="/wasp/scratch/u1657791/cross_match/gaia_dr2/CYCLE1807/YEET"
XMATCHFILENAME="Uncut_Final_YEET.fits"
# HPCLOCATION="/home/jtb34/rds/hpc-work/GACF_OUTPUTS/YEET"
HPCLOCATION="/home/jtb34/rds/rds-jtb34-gacf/GACF_OUTPUTS/YEET"
XMATCHHPCLOCATION="$HPCLOCATION/cross_match"
NGTSFIELDRUN="/home/jtb34/GitHub/GACF/example/hpc/NGTSFieldRun.py"
SBATCHSCRIPT="/home/jtb34/GitHub/GACF/example/hpc/example_submission.peta4-skylake"


if [ "$1" != "" ]; then
    echo "Using fieldname $1"
    FIELDNAME=$1
else
    echo "Enter fieldname (e.g. NG0004-2950):"
    read FIELDNAME
fi

if [ "$3" != "" ]; then
    case $3 in
        [Nn]* ) echo "Not Copying data"; COPYDATA="false";;
        * ) echo "Copying data"; COPYDATA="true";;
    esac
fi

HPCDIR="${HPCLOCATION//YEET/$FIELDNAME}"
HPCXMATCHDIR="${XMATCHHPCLOCATION//YEET/$FIELDNAME}"
#echo $HPCXMATCHDIR

if [ -d $HPCDIR ]
then
    if [ "$COPYDATA" == "true" ]
    then
        if [ "$4" != "" ]; then
            case $4 in
                [Nn]* ) echo "Not Copying data"; COPYDATA="false";;
                * ) echo "Copying data"; COPYDATA="true";;
            esac
        else
            while true; do
                read -p "$HPCDIR already exists on server, continue?" yn
                case $yn in
                    [Yy]* ) break;;
                    [Nn]* ) exit;;
                    * ) echo "Please answer yes or no.";;
                esac
            done
        fi
    fi
fi

mkdir -p "$HPCXMATCHDIR"

#echo $FIELDNAME

NGPHOTGET=$(cat $NGPHOTGETINFO | grep $FIELDNAME)

if [ -z "$NGPHOTGET" ]
then
    echo "No file found for $FIELDNAME"
    exit 1
#else
#    echo "$NGPHOTGET"
fi


USERNAME="jtb34"
XMATCHFILE="${XMATCHLOCATION//YEET/$FIELDNAME}/${XMATCHFILENAME//YEET/$FIELDNAME}"

#echo $XMATCHFILE

if [ "$COPYDATA" == "true" ]
then
    ssh $USERNAME@$NGTSHEAD "test -e $XMATCHFILE"
    if [ $? -eq 0 ]
    then
        echo "$XMATCHFILE exists"
    else
        echo "$XMATCHFILE does not exist"
        exit 1
    fi
fi

cp "$NGTSFIELDRUN" "$HPCDIR"
sed "s/YEET/$FIELDNAME/g" "$SBATCHSCRIPT" > "$HPCDIR/$FIELDNAME.peta4-skylake"

if [ "$COPYDATA" == "true" ]
then
    echo "Copying xmatch data"
    scp ${USERNAME}@${NGTSHEAD}:${XMATCHFILE} ${HPCXMATCHDIR}
    echo "Generating fits file"
    ssh $USERNAME@$NGTSHEAD "cd ${FITSLOCATION}; ${FITSLOCATION}${NGPHOTGET}"

    echo "Finding fits file"
    REMOTEFITS=$(ssh $USERNAME@$NGTSHEAD "ls -1 $FITSLOCATION | grep $FIELDNAME")

    if [ -z "$REMOTEFITS" ]
        then
            echo "No remote file found for $FIELDNAME"
            exit 1
        else
            echo "Copying fits file $REMOTEFITS"
            scp ${USERNAME}@${NGTSHEAD}:${FITSLOCATION}/${REMOTEFITS} ${HPCDIR}
            echo "Removing fits file"
            ssh ${USERNAME}@${NGTSHEAD} "rm ${FITSLOCATION}/${REMOTEFITS}"
    fi
fi



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