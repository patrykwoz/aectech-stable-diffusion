using Autodesk.Revit.UI;
using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Xaml.Behaviors;

namespace StableDiffusionMc.Revit
{
    public class AppCommand : IExternalApplication
    {
        public static AppCommand Instance;

        public Result OnStartup(UIControlledApplication app)
        {
            Instance = this;

            _ = TriggerBase.ActionsProperty;

            try
            {
                app.CreateRibbonTab("Stable Diffusion Mc");
            }
            catch (Exception ex)
            {
                //Logger.Fatal(ex);
            }

            var ribbonPanel = app.GetRibbonPanels("Stable Diffusion Mc").FirstOrDefault(x => x.Name == "Stable Diffusion Mc") ??
                              app.CreateRibbonPanel("Stable Diffusion Mc", $"Stable Diffusion Mc - {typeof(AppCommand).Assembly.GetName().Version}");
            
            StableDiffusionApi.StableDiffusionApiCommand.CreateButton(ribbonPanel);


            return Result.Succeeded;
        }

        public Result OnShutdown(UIControlledApplication app)
        {
            return Result.Succeeded;
        }
    }
}
