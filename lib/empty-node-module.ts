// Browser-only stub for Node built-ins referenced by OpenCV.js in unreachable Node paths.
// PaddleOCR runs only in the browser; the Node fs/path branches are not used at runtime.
const emptyNodeModule = {};
export default emptyNodeModule;
