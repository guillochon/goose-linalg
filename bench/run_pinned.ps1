# Builds the Goose benchmark under several MSVC flag sets and runs it, and
# the OpenBLAS benchmark, each pinned to one performance core so that a
# hybrid CPU (P- and E-cores) does not move them around mid-run.
#
#   powershell -File bench\run_pinned.ps1 [-Core 4] [-Only goose|lapack]
#
# Writes bench_goose_<config>.csv and bench_lapack_1t.csv / _mt.csv here.
param([int]$Core = 4, [string]$Only = "")

$here = $PSScriptRoot
$goose = if ($env:GOOSE) { $env:GOOSE } else { "goose" }
$mask = [IntPtr](1 -shl $Core)

function Run-Pinned($exe, $argList, $log, [bool]$pin = $true) {
    $p = Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory $here -NoNewWindow -PassThru `
        -RedirectStandardOutput $log
    if ($pin) { $p.ProcessorAffinity = $mask }
    $p.WaitForExit()
}

if ($Only -ne "lapack") {
    $configs = [ordered]@{
        "base" = "";                            # what `goose build` does: /O2
        "avx2" = "/arch:AVX2 /fp:contract";     # production: AVX2 + FMA, IEEE semantics kept
        "fast" = "/arch:AVX2 /fp:fast";         # reference only: breaks inf/NaN semantics
    }
    foreach ($name in $configs.Keys) {
        $env:CL = $configs[$name]
        & $goose build -o "$here\bench_$name.exe" "$here\bench.goose" | Out-Null
        Remove-Item Env:CL -ErrorAction SilentlyContinue
        Run-Pinned "$here\bench_$name.exe" @("bench_goose_$name.csv") "$here\log_$name.txt"
        Write-Output "goose $name done"
    }
}
if ($Only -ne "goose") {
    $env:OPENBLAS_NUM_THREADS = "1"
    Run-Pinned "python" @("bench_lapack.py", "lapack_1t") "$here\log_lapack_1t.txt"
    Remove-Item Env:OPENBLAS_NUM_THREADS
    Run-Pinned "python" @("bench_lapack.py", "lapack_mt") "$here\log_lapack_mt.txt" $false   # all cores
    Write-Output "openblas done"
}
