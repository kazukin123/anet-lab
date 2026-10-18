// PRD089: 変更前の手書きSVGと現行の直列Plotly exportを同じ入力で比較する。
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const repo = path.resolve(__dirname, "../../../..");
const output = path.join(repo, "apps/metrics-viewer/target/prd089-validation");
const baselinePath = path.join(output, "baseline-metrics-viewer.js");
const candidatePath = path.join(repo, "apps/metrics-viewer/src/main/resources/static/metrics-viewer.js");
const { chromium } = require(require.resolve("playwright", { paths: [repo, path.resolve(path.dirname(process.execPath), "..")] }));
const quantile = (values, p) => values.slice().sort((a,b)=>a-b)[Math.ceil((values.length-1)*p)];

(async () => {
	if (!fs.existsSync(baselinePath)) throw new Error(`Missing pre-change source: ${baselinePath}`);
	const browser = await chromium.launch({ channel: "msedge", headless: true });
	const rows = [];
	try {
		// 独立したpageを使い、各方式で5回の初回表示・保持メモリを測る。
		for (let repetition=0; repetition<5; repetition++) for (const mode of ["baseline", "candidate"]) {
			const page = await browser.newPage({ viewport:{width:1280,height:720} });
			await page.setContent('<div id="main-area" style="width:900px"></div>');
			await page.addScriptTag({url:"https://cdn.plot.ly/plotly-2.27.0.min.js"});
			const script=fs.readFileSync(mode==="baseline" ? baselinePath : candidatePath,"utf8");
			await page.addScriptTag({content:script + "\nwindow.BenchmarkController=PlotlyController;" + (mode==="candidate" ? "window.BenchmarkRenderer=OverviewSvgRenderer;" : "")});
			const sample=await page.evaluate(async mode => {
				const area=document.getElementById("main-area");
				const colors=["#2F7DE1","#F2C230","#7A5CFF","#008B8B"];
				const traces=Array.from({length:4},(_,run)=>({type:"scatter",mode:"lines",
					name:`run_20261014-${run}_dm_fb_munch_tau3e-2_ndtpen20_nzrep_tf8_100m`,
					x:Array.from({length:192},(_,i)=>i*1000),y:Array.from({length:192},(_,i)=>run*300+(i%3===0?200:1200)+i*2),
					line:{width:1.5,color:colors[run]},meta:{runId:"run-"+run},visible:true,opacity:0.8}));
				const app={logScaleTags:new Set(),_viewportFor:()=>({from:0,to:191000,width:191000}),
					manualYRange:()=>null,percentileBounds:()=>({lower:0,upper:100}),issueForTag:()=>null,combinedStats:()=>null};
				let generatedMs=0, firstMs=null, count=0;
				const start=performance.now();
				if (mode==="baseline") {
					const controller=new window.BenchmarkController(app);
					for(let tag=0;tag<64;tag++) {
						const begin=performance.now(); controller._createPreview(area,null,"tag-"+tag,traces);
						generatedMs+=performance.now()-begin;
						if(firstMs===null) firstMs=performance.now()-start;
					}
				} else {
					const originalPlot=Plotly.newPlot, originalImage=Plotly.toImage;
					Plotly.newPlot=async(...args)=>{const begin=performance.now();const value=await originalPlot(...args);generatedMs+=performance.now()-begin;return value;};
					Plotly.toImage=async(...args)=>{const begin=performance.now();const value=await originalImage(...args);generatedMs+=performance.now()-begin;return value;};
					const renderer=new window.BenchmarkRenderer(tag=>Number(tag.substring(4)),error=>{throw error;});
					window.benchmarkRenderer=renderer;
					const layout={width:900,height:300,margin:{t:30,b:20,l:50,r:10},plot_bgcolor:"#111",paper_bgcolor:"#111",font:{color:"#ccc"},showlegend:true};
					for(let tag=0;tag<64;tag++) renderer.request("tag-"+tag,[tag],{data:traces,layout:{...layout}},()=>true,async url=>{
						const image=document.createElement("img");image.src=url;await image.decode();area.append(image);
						if(firstMs===null) firstMs=performance.now()-start;count++;
					});
					while(count<64 || renderer.running) await new Promise(resolve=>setTimeout(resolve,1));
				}
				return {plotlyVersion:Plotly.version,generatedMs,firstMs,wallMs:performance.now()-start,
					livePlots:[...document.querySelectorAll(".js-plotly-plot")].filter(p=>p.data&&p._fullLayout).length};
			},mode);
			const cdp=await page.context().newCDPSession(page);
			await cdp.send("HeapProfiler.collectGarbage");
			const heap=await cdp.send("Runtime.getHeapUsage");
			rows.push({mode,repetition,...sample,heap});
			await page.close();
		}
	} finally { await browser.close(); }
	const candidates=rows.filter(r=>r.mode==="candidate");
	const baseline=rows.filter(r=>r.mode==="baseline");
	const firstDeltas=candidates.map((row,i)=>row.firstMs-baseline[i].firstMs);
	const memoryDeltas=candidates.map((row,i)=>row.heap.usedSize+row.heap.embedderHeapUsedSize-baseline[i].heap.usedSize-baseline[i].heap.embedderHeapUsedSize);
	const summary={generationMedianMs:quantile(candidates.map(r=>r.generatedMs),0.5),
		generationP95Ms:quantile(candidates.map(r=>r.generatedMs),0.95),
		firstAdditionalP95Ms:quantile(firstDeltas,0.95),retainedAdditionalP95Bytes:quantile(memoryDeltas,0.95)};
	const result={fixture:{graphs:64,runsPerGraph:4,verticesPerRun:192,repetitions:5},
		sourceSha256:Object.fromEntries([["baseline",baselinePath],["candidate",candidatePath]].map(([key,file])=>[key,crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex")])),
		summary,rows,passed:summary.generationP95Ms<=1500&&summary.firstAdditionalP95Ms<=100&&summary.retainedAdditionalP95Bytes<=32*1024*1024&&candidates.every(r=>r.livePlots===0)};
	fs.mkdirSync(output,{recursive:true});
	fs.writeFileSync(path.join(output,"render-performance.json"),JSON.stringify(result,null,2)+"\n");
	console.log(JSON.stringify({summary,passed:result.passed}));
	if(!result.passed) process.exitCode=1;
})().catch(error=>{console.error(error);process.exitCode=1;});
