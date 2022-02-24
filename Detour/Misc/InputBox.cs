using System;
using System.Drawing;
using System.Windows.Forms;

namespace Detour.Misc
{
    public static class InputBox
    {
        private static Form frm = new Form();
        public static string ResultValue;
        private static DialogResult DialogRes;
        private static string[] buttonTextArray = new string[4];
        public enum Icon
        {
            Error,
            Exclamation,
            Information,
            Question,
            Nothing
        }
        public enum Type
        {
            ComboBox,
            TextBox,
            Nothing
        }
        public enum Buttons
        {
            Ok,
            OkCancel,
            YesNo,
            YesNoCancel
        }
        public enum Language
        {
            Czech,
            English,
            German,
            Slovakian,
            Spanish
        }

        public static DialogResult ShowDialog(string Message, string Title = "", string text = "", Buttons buttons = Buttons.Ok)
        {
            DialogRes = DialogResult.Cancel;
            frm.Controls.Clear();
            ResultValue = "";
            //Form definition
            frm.TopMost = true;
            frm.MaximizeBox = false;
            frm.MinimizeBox = false;
            frm.FormBorderStyle = FormBorderStyle.SizableToolWindow;
            frm.Size = new Size(355, 265);
            frm.Text = Title;
            frm.ShowIcon = false;
            frm.FormClosing += frm_FormClosing; 
            frm.AutoScaleMode = AutoScaleMode.Font;
            frm.StartPosition = FormStartPosition.CenterParent;

            //Label definition (message)
            Label label = new Label();
            label.Text = Message;
            label.Size = new Size(245, 50);
            label.Location = new System.Drawing.Point(30, 10); 
            label.TextAlign = ContentAlignment.MiddleLeft;
            frm.Controls.Add(label);
            //Add buttons to the form
            foreach (Button btn in Btns(buttons))
                frm.Controls.Add(btn);
            //Add ComboBox or TextBox to the form
            Control ctrl = Cntrl(text);
            frm.Controls.Add(ctrl);
            //Get automatically cursor to the TextBox
            if (ctrl.Name == "textBox")
                frm.ActiveControl = ctrl;
            frm.ShowDialog();
            if (DialogRes == DialogResult.OK || DialogRes == DialogResult.Yes)
            {
                ResultValue = ctrl.Text;
            }
            else ResultValue = "";
            return DialogRes;
        }
        private static void button_Click(object sender, EventArgs e)
        {
            Button button = (Button)sender;
            switch (button.Name)
            {
                case "Yes":
                    DialogRes = DialogResult.Yes;
                    break;
                case "No":
                    DialogRes = DialogResult.No;
                    break;
                case "Cancel":
                    DialogRes = DialogResult.Cancel;
                    break;
                default:
                    DialogRes = DialogResult.OK;
                    break;
            }
            frm.Close();
        }
        private static void textBox_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Enter)
            {
                DialogRes = DialogResult.OK;
                frm.Close();
            }
        }
        private static void frm_FormClosing(object sender, FormClosingEventArgs e)
        {
#pragma warning disable CS0472 // 由于“DialogResult”类型的值永不等于“DialogResult?”类型的 "null"，该表达式的结果始终为“true”
            if (DialogRes != null) { }
#pragma warning restore CS0472 // 由于“DialogResult”类型的值永不等于“DialogResult?”类型的 "null"，该表达式的结果始终为“true”
            else DialogRes = DialogResult.None;
        }
        private static Button[] Btns(Buttons button, Language lang = Language.English)
        {
            //Buttons field for return
            Button[] returnButtons = new Button[3];
            //Buttons instances
            Button OkButton = new Button();
            Button StornoButton = new Button();
            Button AnoButton = new Button();
            Button NeButton = new Button();
            //Set buttons names and text
            OkButton.Text = "确定";
            OkButton.Name = "OK";
            AnoButton.Text = "是";
            AnoButton.Name = "Yes";
            NeButton.Text = "否";
            NeButton.Name = "No";
            StornoButton.Text = "取消";
            StornoButton.Name = "Cancel";
            //Set buttons position
            switch (button)
            {
                case Buttons.Ok:
                    OkButton.Location = new System.Drawing.Point(250, 175);
                    returnButtons[0] = OkButton;
                    break;
                case Buttons.OkCancel:
                    OkButton.Location = new System.Drawing.Point(170, 175);
                    returnButtons[0] = OkButton;
                    StornoButton.Location = new System.Drawing.Point(250, 175);
                    returnButtons[1] = StornoButton;
                    break;
                case Buttons.YesNo:
                    AnoButton.Location = new System.Drawing.Point(170, 175);
                    returnButtons[0] = AnoButton;
                    NeButton.Location = new System.Drawing.Point(250, 175);
                    returnButtons[1] = NeButton;
                    break;
                case Buttons.YesNoCancel:
                    AnoButton.Location = new System.Drawing.Point(90, 175);
                    returnButtons[0] = AnoButton;
                    NeButton.Location = new System.Drawing.Point(170, 175);
                    returnButtons[1] = NeButton;
                    StornoButton.Location = new System.Drawing.Point(250, 175);
                    returnButtons[2] = StornoButton;
                    break;
            }
            //Set size and event for all used buttons
            foreach (Button btn in returnButtons)
            {
                if (btn != null)
                {
                    btn.Anchor = AnchorStyles.Right | AnchorStyles.Bottom;
                    btn.Size = new Size(75, 30);
                    btn.Click += button_Click;
                }
            }
            return returnButtons;
        }
        private static Control Cntrl(string ListItems)
        {
            //Textbox
            TextBox textBox = new TextBox();
            textBox.Size = new Size(230, 100);
            textBox.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
            textBox.Location = new System.Drawing.Point(40, 60);
            textBox.KeyDown += textBox_KeyDown;
            textBox.Name = "textBox";
            textBox.Text = ListItems;
            textBox.Multiline = true;
            //Set returned Control
            Control returnControl = new Control();
            returnControl = textBox;
            return returnControl;
        }
        public static void SetLanguage(Language lang)
        {
            switch (lang)
            {
                case Language.Czech:
                    buttonTextArray = "OK,Ano,Ne,Storno".Split(',');
                    break;
                case Language.German:
                    buttonTextArray = "OK,Ja,Nein,Stornieren".Split(',');
                    break;
                case Language.Spanish:
                    buttonTextArray = "OK,Sí,No,Cancelar".Split(',');
                    break;
                case Language.Slovakian:
                    buttonTextArray = "OK,Áno,Nie,Zrušit".Split(',');
                    break;
                default:
                    buttonTextArray = "OK,Yes,No,Cancel".Split(',');
                    break;
            }
        }
    }

}
