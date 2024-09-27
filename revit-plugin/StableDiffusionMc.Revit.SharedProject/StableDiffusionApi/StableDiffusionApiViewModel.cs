using System;
using System.Collections.Generic;
using System.Text;
using CommunityToolkit.Mvvm.Input;
using System.Windows;
using StableDiffusionMc.Revit.Core.Utilities.WPF;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using static StableDiffusionMc.Revit.StableDiffusionOnnx.StableDiffusionOnnxModel;
using System.Diagnostics;

namespace StableDiffusionMc.Revit.StableDiffusionApi
{
    public class StableDiffusionApiViewModel: ExtendedViewModelBase
    {
        public StableDiffusionApiModel Model { get; set; }

        public StableDiffusionApiView Win { get; set; }

        public RelayCommand Generate { get; set; }

        public RelayCommand GenerateOnnx { get; set; }

        public string WorkingDirectory { get; set; }

        private string _testImagePath;
        public string TestImagePath
        {
            get { return _testImagePath; }
            set { _testImagePath = value; OnPropertyChanged(); }
        }

        private string _generatedImagePath;

        public string GeneratedImagePath
        {
            get { return _generatedImagePath; }
            set { _generatedImagePath = value; OnPropertyChanged(); }
        }

        private string _prompt= "Enter a prompt...";
        public string Prompt
        {
            get { return _prompt; }
            set { _prompt = value; OnPropertyChanged(); }
        }

        private double _guidanceScale = 7.5;

        public double GuidanceScale
        {
            get { return _guidanceScale; }
            set { _guidanceScale = value; OnPropertyChanged(); }
        }

        private double _strength = 0.85;

        public double Strength
        {
            get { return _strength; }
            set { _strength = value; OnPropertyChanged(); }
        }


        public StableDiffusionApiViewModel(StableDiffusionApiModel model)
        {
            Model = model;

            WindowLoaded = new RelayCommand<Window>(OnWindowLoaded);
            Ok = new RelayCommand<Window>(OnOk);
            Cancel = new RelayCommand<Window>(OnCancel);
            Help = new RelayCommand(OnHelp);
            Generate = new RelayCommand(OnGenerate);
            GenerateOnnx = new RelayCommand(OnGenerateOnnx);

            //var testPath = "C:\\Users\\patry\\Desktop\\stableDifussionBuildingSample.png";
            //GeneratedImages.Add(testPath);
            //TestImagePath = testPath;
        }

        private async void OnGenerate()
        {
            var capturedImagePath = Model.ExportViewAsImagePath();
            if (string.IsNullOrEmpty(capturedImagePath))
            {
                MessageBox.Show("Failed to capture image");
                return;
            }

            var generatedImagePath = await Model.SendToServerAsync(
                capturedImagePath,
                Prompt,
                GuidanceScale,
                Strength
                );

            if (string.IsNullOrEmpty(generatedImagePath))
            {
                MessageBox.Show("Failed to generate image");
                return;
            }

            GeneratedImagePath = generatedImagePath;

        }

        private async void OnGenerateOnnx()
        {
            var capturedImagePath = Model.ExportViewAsImagePath();
            if (string.IsNullOrEmpty(capturedImagePath))
            {
                MessageBox.Show("Failed to capture image");
                return;
            }


            //var options = new SessionOptions();



            string generatedImagePath = null;

            //var basicGenImgPath = await BasicInference(
            //    capturedImagePath,
            //    Prompt,
            //    GuidanceScale,
            //    Strength
            //    );
            Debug.WriteLine("Starting Onnx Inference");
            try
            {
                generatedImagePath = await InferWithOnnx(
                capturedImagePath,
                Prompt,
                GuidanceScale,
                Strength
                );

                Debug.WriteLine("Finished Onnx Inference");

            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
            }
            

            if (string.IsNullOrEmpty(generatedImagePath))
            {
                MessageBox.Show("Failed to generate image");
                return;
            }

            GeneratedImagePath = generatedImagePath;
        }

        public override void OnWindowLoaded(Window win)
        {
            Win = (StableDiffusionApiView)win;

            var capturedImagePath = Model.ExportViewAsImagePath();
            //var testPath = "C:\\Users\\patry\\Desktop\\stableDifussionBuildingSample.png";

            GeneratedImagePath = capturedImagePath;
        }

        public override void OnOk(Window win)
        {
            throw new NotImplementedException();
        }

        public override void OnCancel(Window win)
        {
            Win.Close();
        }

        public override void OnHelp()
        {
            throw new NotImplementedException();
        }
    }
}
