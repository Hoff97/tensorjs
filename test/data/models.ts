export interface ModelData {
  name: string;
  url: string;
  fileName: string;
  zipped: boolean;
  inputData: boolean;
  inputsShape?: number[][];
}


export const models: ModelData[] = [
  {
    name: "ultraface",
    fileName: "version-RFB-320.onnx",
    url: "https://github.com/onnx/models/raw/master/vision/body_analysis/ultraface/models/version-RFB-320.onnx",
    zipped: false,
    inputData: false,
    inputsShape: [[1,3,240,320]]
  },
  {
    name: "mosaic",
    fileName: "mosaic-9.tar.gz",
    url: "https://github.com/onnx/models/raw/master/vision/style_transfer/fast_neural_style/model/mosaic-9.tar.gz",
    zipped: true,
    inputData: true
  }
];