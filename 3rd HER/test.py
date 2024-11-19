from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("\logs")
for i in range(100):
    writer.add_scalar("y=x",i,i)
for i in range(100):
    writer.add_scalar("y=x",2*i,i)
writer.close()