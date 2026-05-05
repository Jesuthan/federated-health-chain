# Run in PowerShell as Administrator
# Downloads and installs IPFS (Kubo) for Windows

$version  = "v0.29.0"
$url      = "https://dist.ipfs.tech/kubo/$version/kubo_${version}_windows-amd64.zip"
$zip      = "$env:TEMP\kubo_windows.zip"
$dest     = "C:\kubo"

Write-Host ""
Write-Host "=== Installing IPFS (Kubo $version) ===" -ForegroundColor Cyan

# Download
Write-Host "Downloading from $url ..."
Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing

# Extract
Write-Host "Extracting to $dest ..."
if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
Expand-Archive -Path $zip -DestinationPath $dest -Force
Remove-Item $zip

# Add to PATH (current session + permanently for user)
$ipfsDir = "$dest\kubo"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$ipfsDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$ipfsDir", "User")
    $env:Path += ";$ipfsDir"
    Write-Host "Added $ipfsDir to PATH" -ForegroundColor Green
}

# Verify
Write-Host ""
& "$ipfsDir\ipfs.exe" version

# Initialise IPFS repo (only needed once)
Write-Host ""
Write-Host "Initialising IPFS repository..."
& "$ipfsDir\ipfs.exe" init

Write-Host ""
Write-Host "IPFS installed. Run 'ipfs daemon' to start." -ForegroundColor Green
