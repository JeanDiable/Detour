using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using DetourCore;

namespace Detour.ToolWindows
{
    public partial class MessageLog : Form
    {
        public MessageLog()
        {
            InitializeComponent();
            listView1.DoubleBuffered(true);
        }

        private void listView1_RetrieveVirtualItem(object sender, RetrieveVirtualItemEventArgs e)
        {
            var n = e.ItemIndex;
            var stat = G.stats.Peek(n+1);
            ListViewItem lvi = new ListViewItem();  // create a listviewitem object
            lvi.Text = stat.Item2.ToString("yy/MM/dd hh:mm:ss");        // assign the text to the item
            ListViewItem.ListViewSubItem lvsi = new ListViewItem.ListViewSubItem();
            lvsi.Text = stat.Item1;
            lvi.SubItems.Add(lvsi);             // assign subitem to item

            e.Item = lvi; 		// assign item to event argument's item-property
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            listView1.VirtualListSize = G.stats.Size();
            listView1.Invalidate();
        }
    }
}
