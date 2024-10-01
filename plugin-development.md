# Revit Plugin Development Guide

## Development Notes

### Important: Set Platform to x64
Ensure the platform is set to `x64`. An architecture mismatch will cause the project to fail at runtime.

### Modify the `.csproj` File
Add the following lines to your `.csproj` file to configure the project properly:

```xml
<Project Sdk="Microsoft.NET.Sdk">
    <PropertyGroup>
        <TargetFramework>net8.0-windows</TargetFramework>
        <UseWPF>true</UseWPF>
        <GenerateAssemblyInfo>true</GenerateAssemblyInfo>
        <GenerateRuntimeConfigurationFiles>true</GenerateRuntimeConfigurationFiles>
        <GenerateDependencyFile>true</GenerateDependencyFile>
        <CopyLocalLockFileAssemblies>true</CopyLocalLockFileAssemblies>
    </PropertyGroup>
    <!-- Add additional configurations here if needed -->
</Project>
```

## Step by Step Guide
Coming soon...