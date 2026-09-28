from client import NormalizationLayers

def main():
    vec = [10.0, 20.0, 30.0, 40.0]
    ln = NormalizationLayers.layer_norm(vec)
    rms = NormalizationLayers.rms_norm(vec)
    print("LayerNorm (zero-mean):", [round(v, 4) for v in ln])
    print("RMSNorm (unit-scale):", [round(v, 4) for v in rms])

if __name__ == "__main__":
    main()
