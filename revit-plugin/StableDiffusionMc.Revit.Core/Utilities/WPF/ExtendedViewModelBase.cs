using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace StableDiffusionMc.Revit.Core.Utilities.WPF
{
    public abstract class ExtendedViewModelBase : ObservableRecipient
    {
        public RelayCommand<Window> WindowLoaded { get; set; }
        public RelayCommand<Window> Ok { get; set; }
        public RelayCommand<Window> Cancel { get; set; }
        public RelayCommand Help { get; set; }

        public abstract void OnWindowLoaded(Window win);
        public abstract void OnOk(Window win);
        public abstract void OnCancel(Window win);
        public abstract void OnHelp();
    }
}
