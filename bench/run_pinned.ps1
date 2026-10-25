# Builds the Goose benchmarks with the C compiler goose finds (clang-cl from
# Visual Studio 2026 on the test machine; GOOSE_CC picks another) and runs
# them, and the OpenBLAS
# benchmark, pinned to performance cores so that a hybrid CPU (P- and
# E-cores) does not move them around mid-run.
#
#   powershell -File bench\run_pinned.ps1 [-Core 4] [-Threads 6] [-ThreadMask 0x555] [-Only goose|lapack|mt]
#
# Single-threaded (n = 50..400), pinned to logical CPU -Core:
#   bench_goose_<config>.csv for three flag sets, bench_lapack_1t.csv
# Multithreaded (n = 400..1600), pinned to -ThreadMask, one logical CPU of
# each performance core (0, 2, ..., 10 on an i5-12600K, whose E-cores are
# 12-15):
#   bench_goose_mt.csv, bench_lapack_mt1.csv (1 thread), bench_lapack_mt.csv
# Unpinned, or on all 16 logical CPUs, OpenBLAS's LU ran 100x slower.
param([int]$Core = 4, [int]$Threads = 6, [long]$ThreadMask = 0x555, [string]$Only = "")

$here = $PSScriptRoot
$goose = if ($env:GOOSE) { $env:GOOSE } else { "goose" }

function Run-Pinned($exe, $argList, $log, [long]$affinity) {
    $p = Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory $here -NoNewWindow -PassThru `
        -RedirectStandardOutput $log
    $p.ProcessorAffinity = [IntPtr]$affinity
    $p.WaitForExit()
}

# Builds a benchmark, failing loudly: a stale executable left by a failed
# build would quietly measure old code.
function Build($src, $exe, $flags) {
    Remove-Item $exe -ErrorAction SilentlyContinue
    $env:CL = $flags
    & $goose build -o $exe $src | Out-Null
    Remove-Item Env:CL -ErrorAction SilentlyContinue
    if (-not (Test-Path $exe)) { throw "building $src failed" }
}

$one = [long](1 -shl $Core)
if ($Only -eq "" -or $Only -eq "goose") {
    $configs = [ordered]@{
        "base" = "";                            # what `goose build` does: /O2, and AVX2 versions of simd fns under clang
        "avx2" = "/arch:AVX2";                  # production: AVX2 throughout, IEEE semantics kept (Goose turns off contraction, so no FMA)
        "fast" = "/arch:AVX2 /fp:fast";         # reference only: FMA under clang, but breaks inf/NaN semantics
    }
    foreach ($name in $configs.Keys) {
        Build "$here\bench.goose" "$here\bench_$name.exe" $configs[$name]
        Run-Pinned "$here\bench_$name.exe" @("bench_goose_$name.csv") "$here\log_$name.txt" $one
        Write-Output "goose $name done"
    }
}
if ($Only -eq "" -or $Only -eq "lapack") {
    $env:OPENBLAS_NUM_THREADS = "1"
    Run-Pinned "python" @("bench_lapack.py", "lapack_1t") "$here\log_lapack_1t.txt" $one
    Remove-Item Env:OPENBLAS_NUM_THREADS
    Write-Output "openblas 1 thread done"
}
if ($Only -eq "" -or $Only -eq "mt") {
    Build "$here\bench_mt.goose" "$here\bench_mt.exe" "/arch:AVX2"
    Run-Pinned "$here\bench_mt.exe" @("bench_goose_mt.csv", "$Threads") "$here\log_mt.txt" $ThreadMask
    Write-Output "goose multithreaded done"
    $env:BENCH_SIZES = "400,800,1600"
    $env:BENCH_OPS = "matmul,det,solve,solve_matrix,inverse,cholesky"
    $env:OPENBLAS_NUM_THREADS = "1"
    Run-Pinned "python" @("bench_lapack.py", "lapack_mt1") "$here\log_lapack_mt1.txt" $one
    $env:OPENBLAS_NUM_THREADS = "$Threads"
    Run-Pinned "python" @("bench_lapack.py", "lapack_mt") "$here\log_lapack_mt.txt" $ThreadMask
    Remove-Item Env:OPENBLAS_NUM_THREADS, Env:BENCH_SIZES, Env:BENCH_OPS
    Write-Output "openblas multithreaded done"
}
