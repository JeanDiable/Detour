
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Resources;

// General Information
[assembly: AssemblyTitle("DetourCore")]
[assembly: AssemblyDescription("Detour SLAM Core")]
[assembly: AssemblyConfiguration("")]
[assembly: AssemblyCompany("Lessokaji")]
[assembly: AssemblyProduct("Lessokaji Detour")]
[assembly: AssemblyCopyright("All rights reserved 2021")]
[assembly: AssemblyTrademark("Lessokaji")]
[assembly: AssemblyCulture("")]

// Version information
[assembly: AssemblyVersion("1.5.0.1424")]
[assembly: NeutralResourcesLanguageAttribute( "en-US" )]

// 1.3.0.0: Todo: use 3D lidar navigation.

// 1.3.1: Update to new SharedObject.
// 1.2.0: TightCoupler done
// 1.2.1: Fix some case when lidar point is sparse, lidar-odometry could fail due to p2l degenerates to p2p.