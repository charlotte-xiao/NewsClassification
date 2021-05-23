import java.awt.*;
import javax.swing.*;
import net.miginfocom.swing.*;
/*
 * Created by JFormDesigner on Fri May 07 18:27:46 CST 2021
 */



/**
 * @author ShenyuanGu
 */
public class 导出新闻内容界面 extends JFrame {
	public 导出新闻内容界面() {
		initComponents();
	}

	private void initComponents() {
		// JFormDesigner - Component initialization - DO NOT MODIFY  //GEN-BEGIN:initComponents
        // Generated using JFormDesigner Evaluation license - unknown
        label1 = new JLabel();
        textField1 = new JTextField();
        label2 = new JLabel();
        scrollPane2 = new JScrollPane();
        list1 = new JList();
        button1 = new JButton();
        button2 = new JButton();

        //======== this ========
        setIconImage(new ImageIcon(getClass().getResource("/static/\u6821\u5fbd.jpg")).getImage());
        var contentPane = getContentPane();
        contentPane.setLayout(new MigLayout(
            "hidemode 3",
            // columns
            "[fill]" +
            "[fill]" +
            "[fill]" +
            "[fill]" +
            "[fill]" +
            "[fill]" +
            "[fill]" +
            "[fill]" +
            "[fill]" +
            "[fill]" +
            "[fill]" +
            "[fill]" +
            "[fill]" +
            "[fill]" +
            "[fill]" +
            "[fill]",
            // rows
            "[]" +
            "[]" +
            "[]" +
            "[]" +
            "[]" +
            "[]" +
            "[]" +
            "[]" +
            "[]" +
            "[]" +
            "[]" +
            "[]" +
            "[]"));

        //---- label1 ----
        label1.setText("\u8f93\u51fa\u5355\u6761\u5185\u5bb9\u7ed3\u679c");
        label1.setForeground(Color.magenta);
        label1.setFont(label1.getFont().deriveFont(label1.getFont().getSize() + 3f));
        contentPane.add(label1, "cell 2 2");
        contentPane.add(textField1, "cell 5 2 10 1");

        //---- label2 ----
        label2.setText("\u8f93\u51fa\u591a\u6761\u5185\u5bb9\u7ed3\u679c");
        label2.setForeground(Color.magenta);
        label2.setFont(label2.getFont().deriveFont(label2.getFont().getSize() + 3f));
        contentPane.add(label2, "cell 2 5");

        //======== scrollPane2 ========
        {
            scrollPane2.setViewportView(list1);
        }
        contentPane.add(scrollPane2, "cell 5 5 10 3");

        //---- button1 ----
        button1.setText("\u786e\u8ba4");
        button1.setForeground(Color.red);
        button1.setFont(button1.getFont().deriveFont(button1.getFont().getSize() + 3f));
        contentPane.add(button1, "cell 2 11");

        //---- button2 ----
        button2.setText("\u5bfc\u51fa");
        button2.setForeground(Color.red);
        button2.setFont(button2.getFont().deriveFont(button2.getFont().getSize() + 3f));
        contentPane.add(button2, "cell 7 11 5 1");
        pack();
        setLocationRelativeTo(getOwner());
		// JFormDesigner - End of component initialization  //GEN-END:initComponents
	}

	// JFormDesigner - Variables declaration - DO NOT MODIFY  //GEN-BEGIN:variables
    // Generated using JFormDesigner Evaluation license - unknown
    private JLabel label1;
    private JTextField textField1;
    private JLabel label2;
    private JScrollPane scrollPane2;
    private JList list1;
    private JButton button1;
    private JButton button2;
	// JFormDesigner - End of variables declaration  //GEN-END:variables
}
