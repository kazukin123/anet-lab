// ---------------- Int32Buffer ----------------
window.Int32Buffer = class {
  constructor(initialSize = 1024) {
    this.buffer = new Int32Array(initialSize);
    this.length = 0;
  }

  _ensureCapacity(minCapacity) {
    if (minCapacity <= this.buffer.length) return;
    let newSize = Math.max(this.buffer.length * 2, minCapacity);
    const newBuffer = new Int32Array(newSize);
    newBuffer.set(this.buffer);
    this.buffer = newBuffer;
  }

  append(values) {
    if (!values || values.length === 0) return;
    const n = values.length;
    const newLen = this.length + n;
    this._ensureCapacity(newLen);
    this.buffer.set(values, this.length);
    this.length = newLen;
  }

  toArray() {
    return this.buffer.subarray(0, this.length);
  }

  clear() {
    this.length = 0;
  }
};

// ---------------- Float32Buffer ----------------
window.Float32Buffer = class {
  constructor(initialSize = 1024) {
    this.buffer = new Float32Array(initialSize);
    this.length = 0;
  }

  _ensureCapacity(minCapacity) {
    if (minCapacity <= this.buffer.length) return;
    let newSize = Math.max(this.buffer.length * 2, minCapacity);
    const newBuffer = new Float32Array(newSize);
    newBuffer.set(this.buffer);
    this.buffer = newBuffer;
  }

  append(values) {
    if (!values || values.length === 0) return;
    const n = values.length;
    const newLen = this.length + n;
    this._ensureCapacity(newLen);
    this.buffer.set(values, this.length);
    this.length = newLen;
  }

  toArray() {
    return this.buffer.subarray(0, this.length);
  }

  clear() {
    this.length = 0;
  }
};
