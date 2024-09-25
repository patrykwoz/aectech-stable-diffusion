using System;
using System.Collections.Generic;
using System.Text;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Windows;
using StableDiffusionMc.Revit.Core.Utilities.WPF;

namespace StableDiffusionMc.Revit.SharedProject.StableDiffusionApi
{
    public class StableDiffusionApiViewModel: ExtendedViewModelBase
    {
        public StableDiffusionApiModel Model { get; set; }

        public RelayCommand<Window> Close { get; set; }




        public StableDiffusionApiViewModel(StableDiffusionApiModel model)
        {
            Model = model;

            //var testPath = "C:\\Users\\patry\\Desktop\\stableDifussionBuildingSample.png";
            //GeneratedImages.Add(testPath);
            //TestImagePath = testPath;

        }

        private void OnCapture()
        {

        }

        private async void OnGenerate()
        {

        }

        private void OnClose(Window win)
        {
            win.Close();
        }

        public override void OnWindowLoaded(Window win)
        {
            throw new NotImplementedException();
        }

        public override void OnOk(Window win)
        {
            throw new NotImplementedException();
        }

        public override void OnCancel(Window win)
        {
            throw new NotImplementedException();
        }

        public override void OnHelp()
        {
            throw new NotImplementedException();
        }
    }
}
