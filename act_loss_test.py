def act_loss_test(outputs):
    batch_size = outputs.size()[0]
    answer = torch.zeros([batch_size,2])
    answer.cuda()
    for i in range(batch_size):
        fake = torch.zeros([1], dtype=torch.float32).to(device)
        real = torch.zeros([1], dtype=torch.float32).to(device)
        # fake latent space
        for latent_index in range(64):
            fake = fake + torch.sum(torch.abs(outputs[i, latent_index]))
        # real latent space
        for latent_index in range(64, 128):
            real = real + torch.sum(torch.abs(outputs[i, latent_index]))

        answer[i][0] = fake.item() / (fake.item() + real.item())

        answer[i][1] = real.item() / (fake.item() + real.item())

    return answer