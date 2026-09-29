# Reflect-and-Revise da rubrica de C5, ponta a ponta: loop na validacao (v0 a v5), escolhe a
# melhor versao, pontua so C5 nas 300 de teste (5 contas em paralelo), junta com o mts_fs2,
# roda evaluate + calibrate. ~1 hora, ~270 chamadas por conta no maximo.
# Se cair no meio, rodar de novo: tudo retoma de onde parou.
$ErrorActionPreference = "Continue"
$proj = $PSScriptRoot
Set-Location $proj
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$log = "results/rr_log_$stamp.txt"

python reflect_revise.py --iters 5 --keys 1,2,3,4,5 *>> $log
$best = python -c "import pandas as pd; h=pd.read_csv('results/rr/historico.csv'); print(int(h.loc[h.qwk_c5.idxmax(),'versao']))"
"melhor versao: v$best" | Tee-Object -FilePath $log -Append

$procs = 0..4 | ForEach-Object {
    $i = $_ + 1
    $off = $_ * 60
    $args = "run_api_scoring.py --provider gemini --model gemini-3.5-flash-lite --modo mts_rr " +
            "--comps C5 --rubrica-c5 data/rubricas_c5/v$best.txt " +
            "--key-env GEMINI_API_KEY_$i --offset $off --limit 60 --rpm 15 " +
            "--out results/api/rrtest_$i.csv"
    Start-Process python -PassThru -WindowStyle Hidden -WorkingDirectory $proj `
        -ArgumentList $args -RedirectStandardError "results/api/log_rrtest_${i}_$stamp.txt"
}
$procs | Wait-Process

python reflect_revise.py --merge-test *>> $log
python evaluate.py results/api/flashlite_mtsrr.csv --confusao *>> $log
python calibrate.py results/api/flashlite_mtsrr.csv --prompts-file results/v5_experimentos_v2/index_prompt_map.csv --out results/calibracao_flashlite_mtsrr.csv *>> $log
"fim. resultado em $log" | Tee-Object -FilePath $log -Append
