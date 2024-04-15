#TODO implement precise error messages

import time as t
import torch

def test_dl(args,train_dl,test_dl):

    """
    Tests if dataloader works as intended

    Args: 
        args: input arguments
        train_dl: dataloader object
        tets_dl dataloader object
        
    Returns:
    """

    # Test dataloader
    if args.debug:

        start = t.time()

        train_samples = 0
        print('test train_dl')
        for train_i,signal in enumerate(train_dl):

            if train_i == 1:
                break

            print('train batch ' + str(train_i))
            print('signal ' + str(signal.shape))
            train_samples += len(signal)

        end = t.time()

        print('Dataloader tested with batch size ' + str(args.batch_size) + '. Time elapsed: ' + str(end-start))

        test_samples = 0
        print('\ntest test_dl')
        for test_i,signal in enumerate(test_dl):

            if test_i == 1:
                break

            test_samples += len(signal)
            print('test batch ' + str(test_i))
            print('signal ' + str(signal.shape))


def test_model(args,device,model,num_patches):

    """
    Tests if model works as intended

    Args: 
        args: input arguments
        device: 
        model: 
        num_patches: number of patches in which the input data is segmented
        
    Returns:
    """

    print('Testing model')

    num_encoder_patches = int(num_patches * (1 - args.tube_mask_ratio))
    num_decoder_patches = int(num_patches * (1 - args.decoder_mask_ratio))

    # test that the model works without error
    model = model.to(device).eval()
    encoder_mask = torch.zeros(num_patches).to(device).to(torch.bool)
    encoder_mask[:num_encoder_patches] = True
    decoder_mask = torch.zeros(num_patches).to(device).to(torch.bool)
    decoder_mask[-num_decoder_patches:] = True

    with torch.no_grad():
        print("\nencoder")
        encoder_out = model(
            torch.randn(2, len(args.bands), 40, 1, 8, 8).to(device),
            encoder_mask=encoder_mask,
            verbose=True,
        )

        print("\ndecoder")
        decoder_out = model(
            encoder_out, encoder_mask=encoder_mask, decoder_mask=decoder_mask, verbose=True
        )

        if args.use_cls_token:
            enc_cls_token = encoder_out[:, :1, :]
            encoder_patches = encoder_out[:, 1:, :]
            dec_cls_token = decoder_out[:, :1, :]
            decoder_patches = decoder_out[:, 1:, :]
            print("enc_cls_token", enc_cls_token.shape)
            print("encoder_patches", encoder_patches.shape)
            print("dec_cls_token", dec_cls_token.shape)
            print("decoder_patches", decoder_patches.shape)