import json
import sys
from client import NormalizationLayers

def handle_request(req):
    method = req.get("method")
    params = req.get("params", {})
    req_id = req.get("id")

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "apply_normalization",
                        "description": "Apply LayerNorm or RMSNorm to vector",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "vector": {"type": "array", "items": {"type": "number"}},
                                "method": {"type": "string", "enum": ["layer_norm", "rms_norm"], "default": "layer_norm"}
                            },
                            "required": ["vector"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "apply_normalization":
            vec = args["vector"]
            m = args.get("method", "layer_norm")
            if m == "rms_norm":
                res = NormalizationLayers.rms_norm(vec)
            else:
                res = NormalizationLayers.layer_norm(vec)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps({"normalized": res})}]}
            }
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
