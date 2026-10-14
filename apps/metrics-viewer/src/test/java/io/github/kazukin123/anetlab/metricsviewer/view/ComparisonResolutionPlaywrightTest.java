package io.github.kazukin123.anetlab.metricsviewer.view;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class ComparisonResolutionPlaywrightTest extends MetricsViewerPlaywrightTestSupport {
	private static final Path WORKSPACES = createFixture();

	@DynamicPropertySource
	static void configure(DynamicPropertyRegistry registry) {
		registry.add("metricsviewer.workspaces-dir", () -> WORKSPACES.toString());
		registry.add("metricsviewer.initial-workspace", () -> "comparison");
		registry.add("metricsviewer.cache-memory-mb", () -> "8");
	}

	@Test
	void fourRunsKeepCommonStepResolutionThroughSelectionZoomPanAndScrolling() {
		page.navigate(baseUrl);
		page.waitForFunction("typeof app !== 'undefined' && ['normal','error'].includes(app.mode)");
		assertEquals("normal", page.evaluate("app.mode"));
		page.evaluate("async () => { await app.refreshMetadata({requestData:false}); }");
		page.waitForFunction("app.cache.getRunIds().length === 4 && app.cache.getRunIds().every(id => app.cache.getRun(id).ingest.state === 'ready')");
		waitForGraph(page);
		page.evaluate("() => Plotly.relayout(document.getElementById(graphId('00/train')), {'xaxis.range':[0,131072]})");
		waitForCommonWidth(1);
		final int initialWidth = width();
		final Object initialVertices=page.evaluate("app.cache.getWindow('run-3','00/train').projection.minMax.steps.length");
		for (int count=2;count<=3;count++) {
			page.evaluate("count => app.setSelectedRuns(Array.from({length:count},(_,i)=>'run-'+(3-i)))",count);
			waitForCommonWidth(count);
			assertEquals(initialWidth,width());
		}
		page.click("#btn-select-all-runs");
		waitForCommonWidth(4);
		assertEquals(initialWidth, width());
		assertEquals(initialVertices,page.evaluate("app.cache.getWindow('run-3','00/train').projection.minMax.steps.length"));
		final int actualBins=((Number)page.evaluate("app.cache.getWindow('run-0','00/train').projection.summary.steps.length")).intValue();
		assertEquals(131056/initialWidth+1,actualBins);
		System.out.println("Comparison resolution: stepWidth="+initialWidth+" denseRunBins="+actualBins+
				" denseRunVertices="+page.evaluate("app.cache.getWindow('run-0','00/train').projection.minMax.steps.length"));
		assertEquals(4, ((Number)page.evaluate("document.getElementById(graphId('00/train')).data.length")).intValue());
		assertTrue(((Number)page.evaluate("document.querySelectorAll('.js-plotly-plot').length")).intValue() < 10);
		assertEquals(64, ((Number)page.evaluate("document.querySelectorAll('.graph-block').length")).intValue());

		page.evaluate("() => Plotly.relayout(document.getElementById(graphId('00/train')), {'xaxis.range':[8000,12000]})");
		waitForCommonWidth(4);
		final int zoomWidth = width();
		assertTrue(zoomWidth < initialWidth);
		final Object overlap = page.evaluate("() => Array.from(app.cache.getWindow('run-0','00/train').projection.summary.steps).filter(x => x>=9000 && x<11000)");
		page.evaluate("() => Plotly.relayout(document.getElementById(graphId('00/train')), {'xaxis.range':[9000,13000]})");
		waitForCommonWidth(4);
		assertEquals(zoomWidth, width());
		assertEquals(overlap, page.evaluate("() => Array.from(app.cache.getWindow('run-0','00/train').projection.summary.steps).filter(x => x>=9000 && x<11000)"));

		// soloでrawまで拡大した後、全Runを戻しても異なる粒度を再利用しない。
		page.evaluate("() => app.setSelectedRuns(['run-0'])");
		waitForCommonWidth(1);
		page.evaluate("() => Plotly.relayout(document.getElementById(graphId('00/train')), {'xaxis.range':[9000,9080]})");
		waitForCommonWidth(1);
		assertEquals(1,width());
		page.click("#btn-select-all-runs");
		waitForCommonWidth(4);
		assertEquals(1,width());
		assertTrue(Boolean.TRUE.equals(page.evaluate("app.selectedRuns.every(id => app.cache.getWindow(id,'00/train').projection.kind === 'raw')")));
		page.evaluate("() => Plotly.relayout(document.getElementById(graphId('00/train')), {'xaxis.range':[9000,13000]})");
		waitForCommonWidth(4);
		assertEquals(zoomWidth,width());

		page.evaluate("() => { document.getElementById('main-area').scrollTop = 15000; }");
		page.waitForFunction("document.getElementById(graphId('49/metric'))?.data?.length > 0");
		assertTrue(((Number)page.evaluate("app.cache.detailBytes()")).longValue() <= 64L*1024*1024);
		page.evaluate("() => { document.getElementById('main-area').scrollTop = 0; }");
		waitForCommonWidth(4);
		assertEquals(zoomWidth, width());
		page.click("#btn-reset-view");
		waitForCommonWidth(4);
		assertTrue(width() > zoomWidth);
		final int beforeResize=width();
		page.setViewportSize(980,720);
		waitForCommonWidth(4);
		assertTrue(width()>beforeResize);
		page.setViewportSize(1280,720);
		waitForCommonWidth(4);
		assertEquals(beforeResize,width());
	}


	@Test
	void scrollingKeepsAllFourRunPreviewsVisibleWhileDetailIsDelayed() {
		page.navigate(baseUrl);
		page.waitForFunction("typeof app !== 'undefined' && app.mode === 'normal'");
		page.evaluate("async () => { await app.refreshMetadata({requestData:false}); }");
		page.waitForFunction("app.cache.getRunIds().length === 4 && app.cache.getRunIds().every(id => app.cache.getRun(id).ingest.state === 'ready')");
		page.evaluate("async () => { app.setSelectedRuns(app.cache.getRunIds()); await app.requestVisibleData(); }");
		assertEquals(1,page.locator(".graph-block[data-tag-key='49/metric'] .graph-log-toggle").count());
		page.waitForFunction("() => {const image=document.querySelector(\".graph-block[data-tag-key='49/metric'] img.graph-preview\"); return image?.complete && image.naturalWidth>0;}");
		assertEquals(4,((Number)page.evaluate("""
				() => {
					const image=document.querySelector(".graph-block[data-tag-key='49/metric'] img.graph-preview");
					const svg=new DOMParser().parseFromString(decodeURIComponent(image.src.substring(image.src.indexOf(',')+1)), 'image/svg+xml');
					if(!svg.querySelector('.ytick') || !svg.querySelector('.xgrid')) throw new Error('Overview axes are missing');
					return svg.querySelectorAll('.legend .traces').length;
				}
				""")).intValue());
		assertEquals(256,((Number)page.evaluate("app.cache.previews.size")).intValue());
		assertEquals(256,((Number)page.evaluate("app.cache.windows.size")).intValue());
		System.out.println("Prefetched detail bytes="+page.evaluate("app.cache.detailBytes()"));
		final var cdp=context.newCDPSession(page);
		System.out.println("Browser heap="+cdp.send("Runtime.getHeapUsage"));
		cdp.detach();
		// 容量内の先読みが完了していれば、スクロール往復にmetricsの再取得は不要。
		final var scrollRequests = new java.util.concurrent.atomic.AtomicInteger();
		page.route("**/api/metrics.json", route -> { scrollRequests.incrementAndGet(); route.resume(); });
		page.locator(".graph-block[data-tag-key='49/metric']").scrollIntoViewIfNeeded();
		page.waitForFunction("document.getElementById(graphId('49/metric'))?.data?.length === 4");
		page.evaluate("document.getElementById('main-area').scrollTop=0");
		page.waitForFunction("document.getElementById(graphId('00/train'))?.data?.length === 4");
		assertEquals(0,scrollRequests.get());
		page.unroute("**/api/metrics.json");

		page.evaluate("""
				() => {
					window.frameBeforeDetail=document.querySelector(".graph-block[data-tag-key='49/metric']");
					window.headerBeforeDetail=window.frameBeforeDetail.querySelector('.graph-header');
				}
				""");
		// 詳細APIを保留しても、スクロールで現れたグラフには概観の線が残る。
		final var held = new java.util.ArrayList<com.microsoft.playwright.Route>();
		page.route("**/api/metrics.json", route -> {
			final var series=com.google.gson.JsonParser.parseString(route.request().postData())
					.getAsJsonObject().getAsJsonArray("series");
			boolean detail=false;
			for(var element:series) {
				final var request=element.getAsJsonObject();
				if(request.get("tagKey").getAsString().equals("49/metric")
						&& request.get("stepBucketWidth").getAsLong()<100) detail=true;
			}
			if(detail) held.add(route); else route.resume();
		});
		page.evaluate("""
				() => {
					const area=document.getElementById('main-area');
					area.style.flex='none'; area.style.width='650px';
					area.scrollTop=document.querySelector(".graph-block[data-tag-key='49/metric']").offsetTop;
					window.dispatchEvent(new Event('resize'));
				}
				""");
		page.waitForCondition(() -> !held.isEmpty());
		assertTrue(page.locator(".graph-block[data-tag-key='49/metric'] .graph-preview").isVisible());
		assertEquals(1,page.locator(".graph-block[data-tag-key='49/metric'] .graph-log-toggle").count());
		page.waitForFunction("() => {const image=document.querySelector(\".graph-block[data-tag-key='49/metric'] img.graph-preview\"); return image?.complete && image.naturalWidth>0;}");
		assertEquals(4,((Number)page.evaluate("""
				() => {
					const image=document.querySelector(".graph-block[data-tag-key='49/metric'] img.graph-preview");
					const svg=new DOMParser().parseFromString(decodeURIComponent(image.src.substring(image.src.indexOf(',')+1)), 'image/svg+xml');
					if(!svg.querySelector('.ytick') || !svg.querySelector('.xgrid')) throw new Error('Overview axes are missing');
					return svg.querySelectorAll('.legend .traces').length;
				}
				""")).intValue());
		assertTrue(((Number)page.evaluate("app.cache.detailBytes()")).longValue()<=64L*1024*1024);
		for(var route:held) route.resume();
		page.waitForFunction("document.getElementById(graphId('49/metric'))?.data?.length === 4");
		assertEquals(0,page.locator(".graph-block[data-tag-key='49/metric'] .graph-preview").count());
		assertEquals(true,page.evaluate("window.frameBeforeDetail===document.querySelector(\".graph-block[data-tag-key='49/metric']\")"));
		assertEquals(true,page.evaluate("window.headerBeforeDetail===document.querySelector(\".graph-block[data-tag-key='49/metric'] .graph-header\")"));
	}

	@Test
	void initialAndMultiRunOverviewsShareAxesAndDisplaySettingsUntilDetailArrives() {
		final var paused = new java.util.ArrayList<com.microsoft.playwright.Route>();
		final var pauseDetail = new java.util.concurrent.atomic.AtomicBoolean(true);
		page.route("**/api/metrics.json",route->{
			final var series=com.google.gson.JsonParser.parseString(route.request().postData()).getAsJsonObject().getAsJsonArray("series");
			boolean detail=false;
			for(var item:series) if(item.getAsJsonObject().get("maxPoints").getAsInt()>198) detail=true;
			if(detail && pauseDetail.get()) paused.add(route); else route.resume();
		});
		page.navigate(baseUrl);
		page.waitForFunction("typeof app !== 'undefined' && app.mode === 'normal'");
		waitForOverview("00/train",1);
		waitForOverview("01/ema",1);
		assertEquals(true,page.evaluate("""
				() => {
					const image=document.querySelector(".graph-block[data-tag-key='01/ema'] img.graph-preview");
					const svg=new DOMParser().parseFromString(decodeURIComponent(image.src.split(',').slice(1).join(',')), 'image/svg+xml');
					return [...svg.querySelectorAll('.ytick text')].every(t=>Number(t.textContent)>0);
				}
				"""));
		page.evaluate("window.initialFrame=document.querySelector(\".graph-block[data-tag-key='00/train']\")");
		page.screenshot(new com.microsoft.playwright.Page.ScreenshotOptions().setPath(Path.of("target/prd089-validation/initial-overview.png")));
		page.click("#btn-select-all-runs");
		waitForOverview("00/train",4);
		waitForOverview("01/ema",4);
		assertEquals(true,page.evaluate("window.initialFrame===document.querySelector(\".graph-block[data-tag-key='00/train']\")"));
		page.screenshot(new com.microsoft.playwright.Page.ScreenshotOptions().setPath(Path.of("target/prd089-validation/multi-run-overview.png")));

		// 概観の共通headerから表示設定を変え、同じPlotlyの軸・Band描画へ反映する。
		page.evaluate("window.previousOverview=document.querySelector(\".graph-block[data-tag-key='01/ema'] img.graph-preview\").src");
		page.locator(".graph-block[data-tag-key='01/ema'] .graph-log-toggle").click();
		page.waitForFunction("document.querySelector(\".graph-block[data-tag-key='01/ema'] img.graph-preview\").src!==window.previousOverview");
		assertEquals("true",page.locator(".graph-block[data-tag-key='01/ema'] .graph-log-toggle").getAttribute("aria-pressed"));
		for(String mode:java.util.List.of("Mean","Band","MinMax")) {
			page.evaluate("window.previousOverview=document.querySelector(\".graph-block[data-tag-key='00/train'] img.graph-preview\").src");
			page.selectOption("#lod-display-mode",mode);
			page.waitForFunction("document.querySelector(\".graph-block[data-tag-key='00/train'] img.graph-preview\").src!==window.previousOverview");
			waitForOverview("00/train",4);
			if(mode.equals("Band")) assertEquals(true,page.evaluate("""
					() => {
						const image=document.querySelector(".graph-block[data-tag-key='00/train'] img.graph-preview");
						const svg=new DOMParser().parseFromString(decodeURIComponent(image.src.split(',').slice(1).join(',')), 'image/svg+xml');
						return svg.querySelectorAll('.fills path').length>=4;
					}
					"""));
		}
		waitForOverview("01/ema",4);
		page.screenshot(new com.microsoft.playwright.Page.ScreenshotOptions().setPath(Path.of("target/prd089-validation/multi-run-overview-log.png")));
		pauseDetail.set(false);
		for(var route:paused) route.resume();
		page.waitForFunction("document.getElementById(graphId('00/train'))?.data?.length===4 && document.getElementById(graphId('01/ema'))?.data?.length===4");
		assertEquals(true,page.evaluate("window.initialFrame===document.querySelector(\".graph-block[data-tag-key='00/train']\")"));
		assertEquals("true",page.locator(".graph-block[data-tag-key='01/ema'] .graph-log-toggle").getAttribute("aria-pressed"));
		page.screenshot(new com.microsoft.playwright.Page.ScreenshotOptions().setPath(Path.of("target/prd089-validation/multi-run-detail.png")));
	}

	@Test
	void lateOverviewImageCannotOverwriteTheLatestRunSelection() {
		final var paused=new java.util.ArrayList<com.microsoft.playwright.Route>();
		page.route("**/api/metrics.json",route->{
			final var series=com.google.gson.JsonParser.parseString(route.request().postData()).getAsJsonObject().getAsJsonArray("series");
			boolean detail=false;
			for(var item:series) if(item.getAsJsonObject().get("maxPoints").getAsInt()>198) detail=true;
			if(detail) paused.add(route); else route.resume();
		});
		page.navigate(baseUrl);
		page.waitForFunction("typeof app !== 'undefined' && app.mode === 'normal'");
		page.waitForFunction("document.querySelectorAll('img.graph-preview').length===64 && [...document.querySelectorAll('img.graph-preview')].every(i=>i.complete && i.naturalWidth>0)");
		page.evaluate("""
				() => {
					const original=Plotly.toImage;
					let pause=true;
					Plotly.toImage=async (...args)=>{
						const url=await original(...args);
						if(pause) {pause=false;await new Promise(resolve=>{window.releaseOverview=resolve;});}
						return url;
					};
				}
				""");
		page.click("#btn-select-all-runs");
		page.waitForFunction("typeof window.releaseOverview==='function'");
		page.click("#btn-latest-only");
		page.waitForFunction("app.selectedRuns.length===1");
		page.evaluate("""
				() => {
					window.publishedLegendCounts=[];
					window.overviewObserver=new MutationObserver(records=>{
						for(const record of records) for(const node of record.addedNodes) {
							if(node.matches?.('img.graph-preview')) {
								const svg=new DOMParser().parseFromString(decodeURIComponent(node.src.substring(node.src.indexOf(',')+1)), 'image/svg+xml');
								window.publishedLegendCounts.push(svg.querySelectorAll('.legend .traces').length);
							}
						}
					});
					window.overviewObserver.observe(document.getElementById('main-area'),{subtree:true,childList:true});
					window.releaseOverview();
				}
				""");
		page.waitForFunction("window.publishedLegendCounts.length>0");
		waitForOverview("00/train",1);
		assertEquals(true,page.evaluate("window.publishedLegendCounts.every(count=>count<=1)"));
		page.evaluate("window.overviewObserver.disconnect()");
		for(var route:paused) route.resume();
	}

	private void waitForOverview(String tagKey, int legends) {
		page.waitForFunction("""
				arg => {
					const image=[...document.querySelectorAll('.graph-block')].find(b=>b.dataset.tagKey===arg.tagKey)?.querySelector('img.graph-preview');
					if(!image?.complete || image.naturalWidth===0) return false;
					const svg=new DOMParser().parseFromString(decodeURIComponent(image.src.substring(image.src.indexOf(',')+1)), 'image/svg+xml');
					return svg.querySelectorAll('.legend .traces').length===arg.legends || (arg.legends===1 && svg.querySelectorAll('.scatterlayer .trace').length===1);
				}
				""",java.util.Map.of("tagKey",tagKey,"legends",legends));
	}

	private int width() {
		return ((Number)page.evaluate("app.cache.getWindow(app.selectedRuns[0], '00/train').stepBucketWidth")).intValue();
	}

	private void waitForCommonWidth(int runs) {
		page.waitForFunction("runs => app.selectedRuns.length === runs && app.selectedRuns.every(id => { const w=app.cache.getWindow(id,'00/train'); return w?.availability==='ok' && w.stepBucketWidth === app._stepBucketWidth('00/train', app._viewportFor('00/train')); })", runs);
	}

	private static Path createFixture() {
		try {
			final Path root = Path.of("target", "comparison-browser-"+System.nanoTime()).toAbsolutePath();
			for (int run=0;run<4;run++) {
				final Path dir=root.resolve("comparison/runs/run-"+run);
				Files.createDirectories(dir);
				final StringBuilder data=new StringBuilder();
				for (int i=0;i<8192;i++) data.append("{\"type\":\"scalar\",\"tag\":\"00/train\",\"step\":")
					.append((i/(run+1))*16).append(",\"value\":").append(i==4321 ? 5000+run*100 : i%97).append("}\n");
				for (int tag=1;tag<64;tag++) for (int i=0;i<16;i++) data.append("{\"type\":\"scalar\",\"tag\":\"")
					.append(tag==1 ? "01/ema" : String.format("%02d/metric",tag)).append("\",\"step\":").append(tag==1 ? (i/(run+1))*1000 : i*1000)
					.append(",\"value\":").append(tag==1 ? 200+run*100+i*50 : i).append("}\n");
				Files.writeString(dir.resolve("metrics.jsonl"),data,StandardCharsets.UTF_8);
			}
			return root;
		} catch (Exception e) { throw new ExceptionInInitializerError(e); }
	}
}
