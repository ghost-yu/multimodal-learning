$ErrorActionPreference = 'Stop'
$source = 'E:/【知识分享】20min入门最优传输.37256235989_原文.srt'
$destination = 'E:/【知识分享】20min入门最优传输.37256235989_中英双语.srt'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$raw = [System.IO.File]::ReadAllText($source, $utf8)
$blocks = [regex]::Split($raw.Trim(), '\r?\n\s*\r?\n')
$translations = @{}
foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $PSScriptRoot 'subtitle_zh.txt'), $utf8)) {
    $parts = $line -split "`t", 2
    if ($parts.Count -ne 2 -or $translations.ContainsKey($parts[0])) { throw "Invalid translation entry: $line" }
    $translations[$parts[0]] = $parts[1]
}
if ($blocks.Count -ne $translations.Count) { throw 'Translation count mismatch' }
$result = foreach ($block in $blocks) {
    $lines = $block -split '\r?\n'
    $id = $lines[0]
    if (-not $translations.ContainsKey($id) -or [string]::IsNullOrWhiteSpace($translations[$id])) { throw "Missing translation: $id" }
    if ($lines[1] -notmatch '^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$') { throw "Invalid timestamp: $id" }
    $english = [regex]::Replace(($lines[2..($lines.Count - 1)] -join ' '), '\p{Cf}', '')
    $id + "`r`n" + $lines[1] + "`r`n" + $translations[$id] + "`r`n" + $english
}
[System.IO.File]::WriteAllText($destination, ($result -join "`r`n`r`n") + "`r`n", [System.Text.UTF8Encoding]::new($true))
$verified = [regex]::Split([System.IO.File]::ReadAllText($destination, $utf8).Trim(), '\r?\n\s*\r?\n')
if ($verified.Count -ne $blocks.Count) { throw 'Output count mismatch' }
for ($i = 0; $i -lt $blocks.Count; $i++) {
    $before = $blocks[$i] -split '\r?\n'
    $after = $verified[$i] -split '\r?\n'
    if ($after.Count -ne 4 -or $after[0] -ne $before[0] -or $after[1] -ne $before[1]) { throw "Output validation failed at $i" }
}
Write-Output "Verified $($verified.Count) bilingual entries; all indices and timestamps preserved."
Write-Output $destination
