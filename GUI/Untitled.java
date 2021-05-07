import java.awt.*;
import javax.swing.*;
import net.miginfocom.swing.*;
/*
 * Created by JFormDesigner on Fri May 07 16:08:27 CST 2021
 */



/**
 * @author ShenyuanGu
 */
public class Untitled extends JFrame {
	public Untitled() {
		initComponents();
	}

	private void initComponents() {
		// JFormDesigner - Component initialization - DO NOT MODIFY  //GEN-BEGIN:initComponents
		// Generated using JFormDesigner Evaluation license - ShenyuanGu
		label1 = new JLabel();
		label5 = new JLabel();
		textField3 = new JTextField();
		label3 = new JLabel();
		textField2 = new JTextField();
		label4 = new JLabel();
		textField4 = new JTextField();
		button1 = new JButton();
		button2 = new JButton();
		button3 = new JButton();

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
			"[]" +
			"[]" +
			"[]" +
			"[]"));

		//---- label1 ----
		label1.setText("\u65b0\u95fb\u6587\u672c\u5206\u7c7b");
		label1.setForeground(Color.red);
		label1.setFont(label1.getFont().deriveFont(label1.getFont().getStyle() & ~Font.ITALIC, label1.getFont().getSize() + 20f));
		contentPane.add(label1, "cell 7 2 6 3");

		//---- label5 ----
		label5.setText("\u7528\u6237\u540d");
		label5.setFont(label5.getFont().deriveFont(label5.getFont().getSize() + 3f));
		contentPane.add(label5, "cell 5 6");
		contentPane.add(textField3, "cell 9 6 5 1");

		//---- label3 ----
		label3.setText("\u5bc6\u7801");
		label3.setFont(label3.getFont().deriveFont(label3.getFont().getSize() + 3f));
		contentPane.add(label3, "cell 5 8");
		contentPane.add(textField2, "cell 9 8 5 1");

		//---- label4 ----
		label4.setText("\u9a8c\u8bc1\u7801");
		label4.setFont(label4.getFont().deriveFont(label4.getFont().getSize() + 3f));
		contentPane.add(label4, "cell 5 10");
		contentPane.add(textField4, "cell 9 10 3 1");

		//---- button1 ----
		button1.setText("\u6ce8\u518c");
		button1.setForeground(Color.blue);
		button1.setFont(button1.getFont().deriveFont(button1.getFont().getSize() + 5f));
		contentPane.add(button1, "cell 5 13 5 3");

		//---- button2 ----
		button2.setText("\u767b\u9646");
		button2.setFont(button2.getFont().deriveFont(button2.getFont().getSize() + 5f));
		button2.setForeground(Color.blue);
		contentPane.add(button2, "cell 11 14");

		//---- button3 ----
		button3.setText("\u4fee\u6539\u5bc6\u7801");
		button3.setForeground(Color.blue);
		button3.setFont(button3.getFont().deriveFont(button3.getFont().getSize() + 5f));
		contentPane.add(button3, "cell 13 14");
		pack();
		setLocationRelativeTo(getOwner());
		// JFormDesigner - End of component initialization  //GEN-END:initComponents
	}

	// JFormDesigner - Variables declaration - DO NOT MODIFY  //GEN-BEGIN:variables
	// Generated using JFormDesigner Evaluation license - ShenyuanGu
	private JLabel label1;
	private JLabel label5;
	private JTextField textField3;
	private JLabel label3;
	private JTextField textField2;
	private JLabel label4;
	private JTextField textField4;
	private JButton button1;
	private JButton button2;
	private JButton button3;
	// JFormDesigner - End of variables declaration  //GEN-END:variables
}
