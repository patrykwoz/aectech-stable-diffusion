using System.Windows.Controls;
using System.Windows.Media;
using System.Windows;

namespace StableDiffusionMc.Revit.StableDiffusionApi
{
    public sealed partial class StableDiffusionApiView
    {
        public StableDiffusionApiView()
        {
            InitializeComponent();
        }

        private void TextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (!(DataContext is StableDiffusionApiViewModel viewModel))
                return;

            if (sender is TextBox textBox)
            {
                switch (textBox.Name)
                {
                    case nameof(PromptTextBox):
                        viewModel.Prompt = textBox.Text;
                        break;
                }
            }
        }

        private void TextBox_GotFocus(object sender, RoutedEventArgs e)
        {
            var textBox = (TextBox)sender;
            if (textBox == null || textBox.Text != "Enter a prompt...")
                return;

            textBox.Text = string.Empty;
            textBox.Foreground = new SolidColorBrush(Colors.Black);
        }

        private void TextBox_LostFocus(object sender, RoutedEventArgs e)
        {
            var textBox = (TextBox)sender;
            if (textBox == null || !string.IsNullOrWhiteSpace(textBox.Text))
                return;

            textBox.Text = "Enter a prompt...";
            textBox.Foreground = new SolidColorBrush(Colors.Gray);
        }
    }
}
