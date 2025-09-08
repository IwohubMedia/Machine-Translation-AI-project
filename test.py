import torch
import argparse
from model import Seq2Seq
from utils import translate_sentence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="How are you?")
    parser.add_argument("--src_lang", type=str, default="en")
    parser.add_argument("--tgt_lang", type=str, default="de")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Seq2Seq()
    model.load_state_dict(torch.load("results/model.pth", map_location=device))
    model.to(device)
    model.eval()

    translation = translate_sentence(args.input, model, args.src_lang, args.tgt_lang, device)
    print(f"Input: {args.input}")
    print(f"Translation: {translation}")
