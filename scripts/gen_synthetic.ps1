# Synthetic Devanagari line-image generator (Windows GDI+).
#
# Why PowerShell and not Python: Pillow on Windows ships without libraqm, so it
# cannot shape Devanagari - it draws the i-matra AFTER the consonant instead of
# before it, producing images that no real page ever contains. GDI+ goes through
# Uniscribe and shapes correctly (verified: "कि" renders with the matra on the
# left, "क्ष" forms the conjunct). Generating wrong-shaped data would train a
# model that fails on every real scan, so this file exists purely for that reason.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/gen_synthetic.ps1 `
#       -Out data/synth/train -Count 2000 -Seed 11
#   powershell -ExecutionPolicy Bypass -File scripts/gen_synthetic.ps1 `
#       -Out data/synth/eval -Count 300 -Seed 99
#
# Output: <Out>/00001.png + <Out>/00001.txt (UTF-8, no BOM)

param(
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Count = 2000,
    [int]$Seed = 11,
    [string]$Corpus = "data/corpus_hi.txt",
    [string]$FontName = "Mangal",
    [int]$Width = 900,
    [int]$Height = 80
)

Add-Type -AssemblyName System.Drawing

if (-not (Test-Path $Corpus)) {
    Write-Error "Corpus not found: $Corpus"
    exit 1
}

# Tokens only; '#' lines are documentation, blank lines are separators.
$tokens = Get-Content -Path $Corpus -Encoding UTF8 |
    Where-Object { $_.Trim() -ne "" -and -not $_.TrimStart().StartsWith("#") } |
    ForEach-Object { $_.Trim() }

if ($tokens.Count -lt 10) {
    Write-Error "Corpus has too few tokens: $($tokens.Count)"
    exit 1
}

New-Item -ItemType Directory -Force -Path $Out | Out-Null

# Deterministic RNG: the same seed must produce the same dataset, otherwise an
# evaluation number cannot be reproduced and is worthless as evidence.
$rng = [System.Random]::new($Seed)
$utf8 = [System.Text.UTF8Encoding]::new($false)

$bitmap = New-Object System.Drawing.Bitmap $Width, $Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.Clear([System.Drawing.Color]::White)
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$graphics.PageUnit = [System.Drawing.GraphicsUnit]::Pixel

$written = 0
$skipped = 0
for ($i = 1; $i -le $Count; $i++) {

    # --- build a line that fits the canvas --------------------------------
    $fontSize = $rng.Next(26, 38)
    $font = New-Object System.Drawing.Font($FontName, $fontSize, [System.Drawing.FontStyle]::Regular)
    $brush = [System.Drawing.Brushes]::Black

    $words = New-Object System.Collections.Generic.List[string]
    $widthUsed = 0.0
    $maxWidth = $Width - 60.0
    $wordTarget = $rng.Next(3, 8)

    while ($words.Count -lt $wordTarget) {
        $word = $tokens[$rng.Next(0, $tokens.Count)]
        # Trailing space is part of the measurement so gaps are accounted for.
        $size = $graphics.MeasureString("$word ", $font)
        if (($widthUsed + $size.Width) -gt $maxWidth) { break }
        $words.Add($word)
        $widthUsed += $size.Width
    }

    if ($words.Count -eq 0) { $font.Dispose(); continue }

    $text = [string]::Join(" ", [string[]]$words.ToArray())

    # 30% of lines end with a danda (U+0964), so the model learns sentence-final
    # punctuation instead of "lines always end in a bare word".
    #
    # Written as a codepoint, NEVER as a literal: Windows PowerShell 5.1 decodes a
    # BOM-less UTF-8 script as ANSI, so a literal danda here becomes "à¥¤" and
    # silently corrupts 30% of the dataset. That is not hypothetical - it happened
    # on the first run of this script.
    if ($rng.NextDouble() -lt 0.30) { $text = $text + [char]0x0964 }

    # Guard: every character must be Devanagari, printable ASCII, or space. Any
    # mojibake from an encoding mistake is dropped here rather than poisoning the
    # training set - a dataset bug is invisible until the model's CER is bad.
    if ($text -notmatch '^[\u0900-\u097F\x20-\x7E]+$') {
        $skipped++
        $font.Dispose()
        continue
    }

    # --- render -----------------------------------------------------------
    $graphics.Clear([System.Drawing.Color]::White)

    $textHeight = $graphics.MeasureString($text, $font).Height
    $x = 24.0 + $rng.Next(0, 12)
    # Small vertical jitter: teaches the model that text is not always perfectly
    # centred, which is the single most common difference between a render and a scan.
    $y = [math]::Max(2.0, ($Height - $textHeight) / 2.0 + $rng.Next(-5, 6))

    $format = New-Object System.Drawing.StringFormat
    $format.FormatFlags = [System.Drawing.StringFormatFlags]::MeasureTrailingSpaces

    $graphics.DrawString($text, $font, $brush, [single]$x, [single]$y, $format)

    $path = Join-Path $Out ("{0:D5}.png" -f $i)
    $bitmap.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    [System.IO.File]::WriteAllText((Join-Path $Out ("{0:D5}.txt" -f $i)), $text, $utf8)

    $format.Dispose()
    $font.Dispose()
    $written++

    if ($i % 250 -eq 0) { Write-Host "  generated $i / $Count" }
}

$graphics.Dispose()
$bitmap.Dispose()

Write-Host "done: $written images in $Out (font=$FontName, seed=$Seed, skipped=$skipped)"
if ($skipped -gt 0) {
    Write-Warning "Skipped $skipped samples with unexpected characters - check script encoding."
}
