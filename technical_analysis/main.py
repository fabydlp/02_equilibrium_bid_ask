from utils import CopelandGalaiCalc

def main():
    print("------- Normal Distribution -------")
    bid_n, ask_n, spread_n = CopelandGalaiCalc.bid_ask_normal(
        mu=102, sigma=7, pi=0.3
    )
    print(f"Bid: {bid_n:.4f}, Ask: {ask_n:.4f}, Spread: {spread_n:.4f}\n")

    print("----- Exponential Distribution -----")
    bid_e, ask_e, spread_e = CopelandGalaiCalc.bid_ask_exponential(
        lam=0.0075, pi=0.01
    )
    print(f"Bid: {bid_e:.4f}, Ask: {ask_e:.4f}, Spread: {spread_e:.4f}")


if __name__ == "__main__":
    main()