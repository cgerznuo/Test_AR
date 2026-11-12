// 카메라 초점거리 자동 추정
// 원리: 인쇄 카드(평면)의 점들과 카메라 영상 속 위치로 호모그래피 H를 구하면,
//       H = K[r1 r2 t] 에서 r1⊥r2, |r1|=|r2| 조건으로 초점거리 f를 역산할 수 있다 (Zhang 캘리브레이션의 1장 버전).
//       카드를 비스듬히 볼 때만 계산이 안정적이므로 기울기 20° 이상인 프레임만 모아 중앙값을 쓴다.

// 정규화 DLT로 호모그래피(3x3, 행 우선) 계산. src/dst: [{x,y}]
function homography(src, dst) {
  const norm = pts => {
    let mx = 0, my = 0; for (const p of pts) { mx += p.x; my += p.y; } mx /= pts.length; my /= pts.length;
    let d = 0; for (const p of pts) d += Math.hypot(p.x - mx, p.y - my); d /= pts.length;
    const s = Math.SQRT2 / (d || 1);
    return {T: [s, 0, -s * mx, 0, s, -s * my, 0, 0, 1], P: pts.map(p => [s * (p.x - mx), s * (p.y - my)])};
  };
  const A = norm(src), B = norm(dst);
  // h33 = 1 고정, 8x8 정규방정식
  const M = Array.from({length: 8}, () => new Float64Array(9));
  const add = (row, rhs) => { for (let i = 0; i < 8; i++) { for (let j = 0; j < 8; j++) M[i][j] += row[i] * row[j]; M[i][8] += row[i] * rhs; } };
  for (let i = 0; i < A.P.length; i++) {
    const [x, y] = A.P[i], [u, v] = B.P[i];
    add([x, y, 1, 0, 0, 0, -u * x, -u * y], u);
    add([0, 0, 0, x, y, 1, -v * x, -v * y], v);
  }
  for (let c = 0; c < 8; c++) {           // 가우스 소거
    let p = c; for (let r = c + 1; r < 8; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    if (Math.abs(M[c][c]) < 1e-12) return null;
    for (let r = 0; r < 8; r++) if (r !== c) { const k = M[r][c] / M[c][c]; for (let j = c; j < 9; j++) M[r][j] -= k * M[c][j]; }
  }
  const h = [...Array(8)].map((_, i) => M[i][8] / M[i][i]).concat(1);
  // H = Tb^-1 * Hn * Ta
  const mul = (a, b) => [0, 1, 2].flatMap(i => [0, 1, 2].map(j => a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]));
  const Tb = B.T, s = Tb[0], TbInv = [1 / s, 0, -Tb[2] / s, 0, 1 / s, -Tb[5] / s, 0, 0, 1];
  return mul(mul(TbInv, h), A.T);
}

// H와 주점(cx,cy)에서 초점거리 f와 카드 기울기(도) 계산
function focalFromH(H, cx, cy) {
  // 주점 이동: H' = [[1,0,-cx],[0,1,-cy],[0,0,1]] H
  const c = j => [H[j] - cx * H[6 + j], H[3 + j] - cy * H[6 + j], H[6 + j]];
  const [a, b, g] = c(0), [d, e, k] = c(1);
  const p1 = a * d + b * e, q1 = g * k;                       // r1·r2 = 0
  const p2 = a * a + b * b - d * d - e * e, q2 = g * g - k * k; // |r1| = |r2|
  const x = -(p1 * q1 + p2 * q2) / (p1 * p1 + p2 * p2);        // x = 1/f²
  if (!(x > 0)) return null;
  const f = 1 / Math.sqrt(x);
  const r1 = [a / f, b / f, g], r2 = [d / f, e / f, k];
  const n1 = Math.hypot(...r1), n2 = Math.hypot(...r2);
  const u = r1.map(v => v / n1), w = r2.map(v => v / n2);
  const r3z = u[0] * w[1] - u[1] * w[0];                        // 카드 법선의 시선 방향 성분
  const tilt = Math.acos(Math.min(1, Math.abs(r3z))) * 180 / Math.PI;
  const ortho = Math.abs(u[0] * w[0] + u[1] * w[1] + u[2] * w[2]); // 잔여 비직교성(품질)
  return {f, tilt, ratio: n1 / n2, ortho};
}

export function attachAutoFocal(mindar, {onChange, onSample} = {}) {
  const ctrl = mindar.controller;
  const W = ctrl.inputWidth, Hh = ctrl.inputHeight, cx = W / 2, cy = Hh / 2;
  const longSide = Math.max(W, Hh);
  const fovLong = f => 2 * Math.atan(longSide / 2 / f) * 180 / Math.PI;
  const samples = [];
  let frame = 0, applied = ctrl.projectionTransform[0][0], lastApply = 0;
  ctrl.onCorrespondences = (world, screen) => {
    if (++frame % 2) return;                          // 2프레임마다
    const H = homography(world, screen); if (!H) return;
    const r = focalFromH(H, cx, cy); if (!r) return;
    const fov = fovLong(r.f);
    onSample && onSample(r, fov);
    if (r.tilt < 20 || fov < 35 || fov > 100 || Math.abs(r.ratio - 1) > 0.15) return;
    samples.push(r.f); if (samples.length > 40) samples.shift();
    if (samples.length < 12) return;
    const med = [...samples].sort((p, q) => p - q)[samples.length >> 1];
    const now = performance.now();
    if (Math.abs(med - applied) / applied > 0.03 && now - lastApply > 1500) {
      applied = med; lastApply = now;
      ctrl.setFocal(med);
      mindar.resize();                                // 3D 카메라 화각도 갱신
      onChange && onChange(fovLong(med), med);
    }
  };
  return {get fovLong() { return fovLong(applied); }, get samples() { return samples.length; }};
}
