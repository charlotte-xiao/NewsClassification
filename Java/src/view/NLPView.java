/*
 * Created by JFormDesigner on Sun May 23 11:59:28 CST 2021
 */

package view;

import com.formdev.flatlaf.FlatIntelliJLaf;
import utils.http;

import javax.swing.*;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

/**
 * @author charlottexiao
 */
public class NLPView extends JFrame {
    public NLPView() {
        FlatIntelliJLaf.install();
        initComponents();
    }

    private void button1MouseClicked(MouseEvent e) {
        // TODO add your code here
        String title = textField.getText();
        String content = textPane.getText();
        String result =http.sendPost("http://127.0.0.1:80/single","title="+title+"&"+"content="+content);
        JOptionPane.showMessageDialog(null, "分类结果:"+result, "结果",JOptionPane.INFORMATION_MESSAGE);
    }

    private void button2MouseClicked(MouseEvent e) {
        // TODO add your code here
    }

    private void initComponents() {
        // JFormDesigner - Component initialization - DO NOT MODIFY  //GEN-BEGIN:initComponents
        // Generated using JFormDesigner Evaluation license - unknown
        panel1 = new JPanel();
        panel2 = new JPanel();
        label = new JLabel();
        panel4 = new JPanel();
        panel5 = new JPanel();
        label2 = new JLabel();
        textField = new JTextField();
        textPane = new JTextPane();
        panel3 = new JPanel();
        button1 = new JButton();
        button2 = new JButton();

        //======== this ========
        setTitle("NewsClassification");
        setIconImage(new ImageIcon(getClass().getResource("/static/ico.jpg")).getImage());
        setResizable(false);
        setMinimumSize(null);
        setName("frame");
        setFont(new Font(Font.DIALOG, Font.PLAIN, 18));
        setVisible(true);
        Container contentPane = getContentPane();
        contentPane.setLayout(new BorderLayout(3, 3));

        //======== panel1 ========
        {
            panel1.setPreferredSize(new Dimension(600, 440));
            panel1.setMinimumSize(new Dimension(0, 0));
            panel1.setBorder ( new javax . swing. border .CompoundBorder ( new javax . swing. border .TitledBorder ( new javax
            . swing. border .EmptyBorder ( 0, 0 ,0 , 0) ,  "JF\u006frmDes\u0069gner \u0045valua\u0074ion" , javax. swing
            .border . TitledBorder. CENTER ,javax . swing. border .TitledBorder . BOTTOM, new java. awt .
            Font ( "D\u0069alog", java .awt . Font. BOLD ,12 ) ,java . awt. Color .red
            ) ,panel1. getBorder () ) ); panel1. addPropertyChangeListener( new java. beans .PropertyChangeListener ( ){ @Override
            public void propertyChange (java . beans. PropertyChangeEvent e) { if( "\u0062order" .equals ( e. getPropertyName (
            ) ) )throw new RuntimeException( ) ;} } );
            panel1.setLayout(new BorderLayout(5, 5));

            //======== panel2 ========
            {
                panel2.setPreferredSize(new Dimension(600, 390));
                panel2.setMinimumSize(null);
                panel2.setLayout(new BorderLayout(5, 5));

                //---- label ----
                label.setFont(new Font("JetBrains Mono", Font.BOLD, 18));
                label.setHorizontalAlignment(SwingConstants.CENTER);
                label.setMaximumSize(null);
                label.setMinimumSize(null);
                label.setPreferredSize(new Dimension(600, 100));
                label.setIcon(new ImageIcon(getClass().getResource("/static/background.png")));
                label.setBackground(Color.white);
                panel2.add(label, BorderLayout.NORTH);

                //======== panel4 ========
                {
                    panel4.setMinimumSize(null);
                    panel4.setPreferredSize(new Dimension(600, 290));
                    panel4.setLayout(new BorderLayout(10, 10));

                    //======== panel5 ========
                    {
                        panel5.setPreferredSize(new Dimension(600, 40));
                        panel5.setLayout(new BorderLayout());

                        //---- label2 ----
                        label2.setText("Title");
                        label2.setFocusable(false);
                        label2.setPreferredSize(new Dimension(150, 40));
                        label2.setHorizontalAlignment(SwingConstants.CENTER);
                        label2.setFont(new Font("JetBrains Mono", Font.PLAIN, 16));
                        label2.setMinimumSize(null);
                        label2.setMaximumSize(null);
                        panel5.add(label2, BorderLayout.LINE_START);

                        //---- textField ----
                        textField.setPreferredSize(new Dimension(450, 40));
                        textField.setMinimumSize(null);
                        textField.setMaximumSize(null);
                        textField.setHorizontalAlignment(SwingConstants.LEFT);
                        panel5.add(textField, BorderLayout.CENTER);
                    }
                    panel4.add(panel5, BorderLayout.NORTH);

                    //---- textPane ----
                    textPane.setPreferredSize(new Dimension(600, 250));
                    textPane.setBorder(new LineBorder(Color.lightGray, 3, true));
                    panel4.add(textPane, BorderLayout.CENTER);
                }
                panel2.add(panel4, BorderLayout.CENTER);
            }
            panel1.add(panel2, BorderLayout.CENTER);

            //======== panel3 ========
            {
                panel3.setPreferredSize(new Dimension(600, 40));
                panel3.setLayout(new GridLayout(1, 2, 15, 15));

                //---- button1 ----
                button1.setText("\u6267\u884c");
                button1.setPreferredSize(null);
                button1.setActionCommand("\u5904\u7406");
                button1.setMinimumSize(new Dimension(350, 10));
                button1.setMaximumSize(new Dimension(350, 10));
                button1.addMouseListener(new MouseAdapter() {
                    @Override
                    public void mouseClicked(MouseEvent e) {
                        button1MouseClicked(e);
                    }
                });
                panel3.add(button1);

                //---- button2 ----
                button2.setText("\u8fd4\u56de");
                button2.setPreferredSize(null);
                button2.setMaximumSize(new Dimension(98, 10));
                button2.setMinimumSize(new Dimension(98, 10));
                button2.addMouseListener(new MouseAdapter() {
                    @Override
                    public void mouseClicked(MouseEvent e) {
                        button2MouseClicked(e);
                    }
                });
                panel3.add(button2);
            }
            panel1.add(panel3, BorderLayout.SOUTH);
        }
        contentPane.add(panel1, BorderLayout.CENTER);
        pack();
        setLocationRelativeTo(getOwner());
        // JFormDesigner - End of component initialization  //GEN-END:initComponents
    }

    // JFormDesigner - Variables declaration - DO NOT MODIFY  //GEN-BEGIN:variables
    // Generated using JFormDesigner Evaluation license - unknown
    private JPanel panel1;
    private JPanel panel2;
    private JLabel label;
    private JPanel panel4;
    private JPanel panel5;
    private JLabel label2;
    private JTextField textField;
    private JTextPane textPane;
    private JPanel panel3;
    private JButton button1;
    private JButton button2;
    // JFormDesigner - End of variables declaration  //GEN-END:variables
}
