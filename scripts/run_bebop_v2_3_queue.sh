#!/usr/bin/env bash

# ============================================================
# Campaña MapNet - Bebop v2_3
# Ejecuta cinco experimentos secuencialmente.
# ============================================================

set -u

REPO_ROOT="/data/research/mapnet"
cd "$REPO_ROOT" || exit 1

PYTHON="$(command -v python)"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="scripts/logs/bebop_v2_3_campaign_${TIMESTAMP}"

mkdir -p "$LOG_DIR"

CAMPAIGN_LOG="$LOG_DIR/campaign.log"

CONFIGS=(
    "configs/Bebop/v2_3/bebop_mapnet_queue_smoke.ini"
    "configs/Bebop/v2_3/bebop_mapnet_baseline.ini"
    "configs/Bebop/v2_3/bebop_mapnet_500.ini"
    "configs/Bebop/v2_3/bebop_mapnet_lr5e-5.ini"
    "configs/Bebop/v2_3/bebop_mapnet_wd1e-4.ini"
    "configs/Bebop/v2_3/bebop_mapnet_wd1e-4_2000.ini"
)

echo "============================================================" | tee -a "$CAMPAIGN_LOG"
echo "CAMPAÑA MAPNET - BEBOP v2_3" | tee -a "$CAMPAIGN_LOG"
echo "Inicio: $(date)" | tee -a "$CAMPAIGN_LOG"
echo "Python: $PYTHON" | tee -a "$CAMPAIGN_LOG"
echo "Logs: $LOG_DIR" | tee -a "$CAMPAIGN_LOG"
echo "============================================================" | tee -a "$CAMPAIGN_LOG"

# Verificar entorno
"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA no disponible'; print('GPU:', torch.cuda.get_device_name(0))" \
    || exit 1

# Verificar configuraciones
for CONFIG in "${CONFIGS[@]}"; do
    if [[ ! -f "$CONFIG" ]]; then
        echo "ERROR: No existe $CONFIG" | tee -a "$CAMPAIGN_LOG"
        exit 1
    fi
done

TOTAL=${#CONFIGS[@]}
EXPERIMENT=0

for CONFIG in "${CONFIGS[@]}"; do

    EXPERIMENT=$((EXPERIMENT + 1))

    NAME="$(basename "$CONFIG" .ini)"
    EXP_LOG="$LOG_DIR/${NAME}.log"

    echo "" | tee -a "$CAMPAIGN_LOG"
    echo "============================================================" | tee -a "$CAMPAIGN_LOG"
    echo "EXPERIMENTO $EXPERIMENT / $TOTAL: $NAME" | tee -a "$CAMPAIGN_LOG"
    echo "Inicio: $(date)" | tee -a "$CAMPAIGN_LOG"
    echo "Configuración: $CONFIG" | tee -a "$CAMPAIGN_LOG"
    echo "Log individual: $EXP_LOG" | tee -a "$CAMPAIGN_LOG"
    echo "============================================================" | tee -a "$CAMPAIGN_LOG"

    START_TIME=$(date +%s)

    "$PYTHON" geomapnet/scripts/train.py \
        --dataset Bebop \
        --scene v2_3 \
        --config_file "$CONFIG" \
        --model mapnet \
        --device 0 \
        --learn_beta \
        --learn_gamma \
        2>&1 | tee "$EXP_LOG"

    STATUS=${PIPESTATUS[0]}

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [[ $STATUS -ne 0 ]]; then
        echo "ERROR en $NAME. Código de salida: $STATUS" | tee -a "$CAMPAIGN_LOG"
        echo "La cola se detiene para evitar ejecutar experimentos posteriores." | tee -a "$CAMPAIGN_LOG"
        echo "Fin: $(date)" | tee -a "$CAMPAIGN_LOG"
        exit "$STATUS"
    fi

    echo "COMPLETADO: $NAME" | tee -a "$CAMPAIGN_LOG"
    echo "Duración: $((DURATION / 3600)) h $(((DURATION % 3600) / 60)) min $((DURATION % 60)) s" | tee -a "$CAMPAIGN_LOG"

done

echo "" | tee -a "$CAMPAIGN_LOG"
echo "============================================================" | tee -a "$CAMPAIGN_LOG"
echo "CAMPAÑA COMPLETADA" | tee -a "$CAMPAIGN_LOG"
echo "Fin: $(date)" | tee -a "$CAMPAIGN_LOG"
echo "============================================================" | tee -a "$CAMPAIGN_LOG"
