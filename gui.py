from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import torch
import model
from data.hw2_labels_dictionary import classes
from torchvision.transforms import transforms

root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

data_transform = transforms.Compose([
    transforms.Resize((30, 30)),
    transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214),
                         (0.2724, 0.2608, 0.2669))
])


def get_prediction(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # trained_model = torch.load("models/best_model_f_conv.pt", map_location=device)
    data_tensor = torch.zeros((64, 3, 30, 30)).to(device)
    fe_dims = [3, 64, 64, 128]
    c_dims = [120, 80, 43]
    trained_model = model.ConvNet(fe_dims=fe_dims, c_dims=c_dims, mode='f_conv').to(device)
    trained_model.load_state_dict(torch.load("final_models/best_model_f_conv.pt", map_location=device))
    img = data_transform(img)
    img = img.to(device)
    data_tensor[0] = img
    y_hat = trained_model(data_tensor)
    y_hat = torch.squeeze(y_hat)
    predictions = torch.argmax(y_hat, dim=1)
    return classes[int(predictions[0].cpu().detach()) + 1]


def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename


def open_img():
    x = openfn()
    img = Image.open(x)
    title = get_prediction(img)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    print(title)
    panel = Label(root, text=title, image=img)
    panel.image = img
    panel.text = title
    text = Text(root)
    text.insert(INSERT, "Predicted: ")
    text.insert(END, title)
    panel.pack()
    text.pack()


btn = Button(root, text='load image', command=open_img).pack()

root.mainloop()
