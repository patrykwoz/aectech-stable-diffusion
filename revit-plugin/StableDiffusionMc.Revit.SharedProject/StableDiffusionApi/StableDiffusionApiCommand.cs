using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Text;
using System.Windows.Interop;
using StableDiffusionMc.Revit.Core.Utilities;

namespace StableDiffusionMc.Revit.StableDiffusionApi
{
    [Transaction(TransactionMode.Manual)]
    [Regeneration(RegenerationOption.Manual)]
    [Journaling(JournalingMode.NoCommandData)]
    public class StableDiffusionApiCommand : IExternalCommand
    {

        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            try
            {
                var uiApp = commandData.Application;

                var m = new StableDiffusionApiModel(uiApp);
                var vm = new StableDiffusionApiViewModel(m);
                var v = new StableDiffusionApiView
                {
                    DataContext = vm
                };

                _ = new WindowInteropHelper(v)
                {
                    Owner = Process.GetCurrentProcess().MainWindowHandle
                };

                v.ShowDialog();



            }
            catch (Exception ex)
            {
                //ignored for now
            }

            return Result.Succeeded;
        }

        public static void CreateButton(RibbonPanel panel)
        {
            var assembly = Assembly.GetExecutingAssembly();
            panel.AddItem(
                new PushButtonData(
                    MethodBase.GetCurrentMethod()?.DeclaringType?.Name,
                    "Stable Diffusion" + Environment.NewLine + "Img2Img",
                    assembly.Location,
                    MethodBase.GetCurrentMethod()?.DeclaringType?.FullName)
                {
                    ToolTip = "Generate images using Stable Diffusion",
                    LargeImage = ImageUtils.LoadImage(assembly, "_32x32.lightning.png"),
                });
        }
    }
}
