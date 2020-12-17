import smtplib
from string import Template
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
FROM_EMAIL = 'githubreporter@zohomail.com'
MY_PASSWORD = 'quang7699'
TO_EMAIL = 'hocmai6@gmail.com,tnquang.769@gmail.com,qq706712@gmail.com'
def parse_template(file_name):
    with open(file_name, 'r', encoding='utf-8') as msg_template:
        msg_template_content = msg_template.read()
    return Template(msg_template_content)

def sendReport(epoch,traintime, loss,wer,cer, lr, note):
      today = date.today()
      datee = today.strftime("%d/%m/%Y")
      message_template = parse_template('/work/Source/deepspeech.pytorch/deepspeech_pytorch/mailtemp.html')
      smtp_server = smtplib.SMTP_SSL('smtp.zoho.com', 465)
      smtp_server.login('tnquang.769@gmail.com', 'quang7699')
      multipart_msg = MIMEMultipart() 
      message = message_template.substitute(
            EPOCH_STT=epoch,
            traintime =traintime,
            loss = loss,wer=wer     ,cer=cer,lr=lr,
            note = note
            )
      multipart_msg['From']=FROM_EMAIL
      multipart_msg['To']= TO_EMAIL
      multipart_msg['Subject']= str(datee)+ " REPORT SUMMARY EPOCH : "+str(epoch)
      multipart_msg.attach(MIMEText(message, 'html'))
      smtp_server.send_message(multipart_msg)
      del multipart_msg
      smtp_server.quit()  
