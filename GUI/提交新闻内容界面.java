import java.awt.*;
import javax.swing.*;
import net.miginfocom.swing.*;
/*
 * Created by JFormDesigner on Fri May 07 17:54:08 CST 2021
 */



/**
 * @author ShenyuanGu
 */
public class 提交新闻内容界面 extends JFrame {
	public 提交新闻内容界面() {
		initComponents();
	}

	private void initComponents() {
		// JFormDesigner - Component initialization - DO NOT MODIFY  //GEN-BEGIN:initComponents
		// Generated using JFormDesigner Evaluation license - ShenyuanGu
		textField1 = new JTextField();
		label1 = new JLabel();
		label2 = new JLabel();
		scrollPane1 = new JScrollPane();
		list1 = new JList();
		button1 = new JButton();
		button2 = new JButton();

		//======== this ========
		Container contentPane = getContentPane();
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
			"[]"));
		contentPane.add(textField1, "cell 10 2 9 3");

		//---- label1 ----
		label1.setText("\u8f93\u5165\u5355\u6761\u65b0\u95fb\u5185\u5bb9");
		label1.setForeground(Color.magenta);
		contentPane.add(label1, "cell 5 3");

		//---- label2 ----
		label2.setText("\u8f93\u5165\u6279\u91cf\u65b0\u95fb\u5185\u5bb9");
		label2.setForeground(Color.magenta);
		contentPane.add(label2, "cell 5 5");

		//======== scrollPane1 ========
		{
			scrollPane1.setViewportView(list1);
		}
		contentPane.add(scrollPane1, "cell 10 5 9 1");

		//---- button1 ----
		button1.setText("\u63d0\u4ea4");
		button1.setForeground(Color.red);
		button1.setFont(button1.getFont().deriveFont(button1.getFont().getSize() + 5f));
		contentPane.add(button1, "cell 5 8");

		//---- button2 ----
		button2.setText("\u8fd4\u56de");
		button2.setForeground(Color.red);
		button2.setFont(button2.getFont().deriveFont(button2.getFont().getSize() + 5f));
		contentPane.add(button2, "cell 10 8 4 1");
		pack();
		setLocationRelativeTo(getOwner());
		// JFormDesigner - End of component initialization  //GEN-END:initComponents
	}

	// JFormDesigner - Variables declaration - DO NOT MODIFY  //GEN-BEGIN:variables
	// Generated using JFormDesigner Evaluation license - ShenyuanGu
	private JTextField textField1;
	private JLabel label1;
	private JLabel label2;
	private JScrollPane scrollPane1;
	private JList list1;
	private JButton button1;
	private JButton button2;
	// JFormDesigner - End of variables declaration  //GEN-END:variables
}
