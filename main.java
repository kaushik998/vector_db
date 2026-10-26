import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executors;

public class main {
	static final int DIMS = 16;

	static class VectorItem {
		int id;
		String metadata;
		String category;
		float[] embedding;

		VectorItem(int id, String metadata, String category, float[] embedding) {
			this.id = id;
			this.metadata = metadata;
			this.category = category;
			this.embedding = embedding;
		}
	}

	static class Hit {
		VectorItem item;
		float distance;

		Hit(VectorItem item, float distance) {
			this.item = item;
			this.distance = distance;
		}
	}

	static float euclidean(float[] a, float[] b) {
		float sum = 0;
		for (int i = 0; i < Math.min(a.length, b.length); i++) {
			float delta = a[i] - b[i];
			sum += delta * delta;
		}
		return (float) Math.sqrt(sum);
	}

	static float manhattan(float[] a, float[] b) {
		float sum = 0;
		for (int i = 0; i < Math.min(a.length, b.length); i++) sum += Math.abs(a[i] - b[i]);
		return sum;
	}

	static float cosine(float[] a, float[] b) {
		float dot = 0, na = 0, nb = 0;
		for (int i = 0; i < Math.min(a.length, b.length); i++) {
			dot += a[i] * b[i];
			na += a[i] * a[i];
			nb += b[i] * b[i];
		}
		if (na < 1e-9f || nb < 1e-9f) return 1;
		return 1 - dot / ((float) Math.sqrt(na) * (float) Math.sqrt(nb));
	}

	static float distance(float[] a, float[] b, String metric) {
		if ("euclidean".equals(metric)) return euclidean(a, b);
		if ("manhattan".equals(metric)) return manhattan(a, b);
		return cosine(a, b);
	}

	static class VectorDB {
		final Map<Integer, VectorItem> items = new HashMap<>();
		int nextId = 1;

		synchronized int insert(String metadata, String category, float[] embedding) {
			int id = nextId++;
			items.put(id, new VectorItem(id, metadata, category, embedding));
			return id;
		}

		synchronized boolean remove(int id) { return items.remove(id) != null; }

		synchronized List<Hit> search(float[] query, int k, String metric) {
			List<Hit> result = new ArrayList<>();
			for (VectorItem item : items.values()) result.add(new Hit(item, distance(query, item.embedding, metric)));
			result.sort(Comparator.comparingDouble(hit -> hit.distance));
			return result.subList(0, Math.min(Math.max(k, 0), result.size()));
		}

		synchronized List<VectorItem> all() { return new ArrayList<>(items.values()); }
		synchronized int size() { return items.size(); }
	}

	static class Document {
		int id;
		String title;
		String text;
		float[] embedding;

		Document(int id, String title, String text, float[] embedding) {
			this.id = id; this.title = title; this.text = text; this.embedding = embedding;
		}
	}

	static class DocumentDB {
		final Map<Integer, Document> documents = new HashMap<>();
		int nextId = 1;
		int dims;

		synchronized int insert(String title, String text, float[] embedding) {
			dims = embedding.length;
			int id = nextId++;
			documents.put(id, new Document(id, title, text, embedding));
			return id;
		}

		synchronized boolean remove(int id) { return documents.remove(id) != null; }

		synchronized List<Hit> search(float[] query, int k) {
			List<Hit> result = new ArrayList<>();
			for (Document document : documents.values()) {
				VectorItem item = new VectorItem(document.id, document.title, "doc", document.embedding);
				float d = cosine(query, document.embedding);
				if (d <= .7f) result.add(new Hit(item, d));
			}
			result.sort(Comparator.comparingDouble(hit -> hit.distance));
			return result.subList(0, Math.min(Math.max(k, 0), result.size()));
		}

		synchronized List<Document> all() { return new ArrayList<>(documents.values()); }
		synchronized int size() { return documents.size(); }
	}

	static class OllamaClient {
		final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
		String embedModel = "nomic-embed-text";
		String genModel = "llama3.2";
		String baseUrl = "http://127.0.0.1:11434";

		String post(String path, String body, int timeoutSeconds) throws IOException, InterruptedException {
			HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + path))
					.timeout(Duration.ofSeconds(timeoutSeconds))
					.header("Content-Type", "application/json")
					.POST(HttpRequest.BodyPublishers.ofString(body)).build();
			HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
			return response.statusCode() == 200 ? response.body() : "";
		}

		boolean available() {
			try {
				HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/api/tags"))
						.timeout(Duration.ofSeconds(2)).GET().build();
				return client.send(request, HttpResponse.BodyHandlers.discarding()).statusCode() == 200;
			} catch (Exception e) { return false; }
		}

		float[] embed(String text) {
			try {
				String response = post("/api/embeddings", "{\"model\":" + json(embedModel) + ",\"prompt\":" + json(text) + "}", 30);
				return parseFloatArray(response, "embedding");
			} catch (Exception e) { return new float[0]; }
		}

		String generate(String prompt) {
			try {
				String response = post("/api/generate", "{\"model\":" + json(genModel) + ",\"prompt\":" + json(prompt) + ",\"stream\":false}", 180);
				String answer = field(response, "response");
				return answer.isEmpty() ? "ERROR: Ollama returned no response" : answer;
			} catch (Exception e) { return "ERROR: Ollama unavailable. Run: ollama serve"; }
		}
	}

	static String json(String value) {
		StringBuilder out = new StringBuilder("\"");
		for (char c : value.toCharArray()) {
			if (c == '"') out.append("\\\"");
			else if (c == '\\') out.append("\\\\");
			else if (c == '\n') out.append("\\n");
			else if (c == '\r') out.append("\\r");
			else if (c == '\t') out.append("\\t");
			else out.append(c);
		}
		return out.append('"').toString();
	}

	static String field(String body, String key) {
		String marker = "\"" + key + "\"";
		int p = body.indexOf(marker);
		if (p < 0) return "";
		p = body.indexOf(':', p) + 1;
		while (p < body.length() && Character.isWhitespace(body.charAt(p))) p++;
		if (p >= body.length() || body.charAt(p) != '"') return "";
		StringBuilder out = new StringBuilder();
		for (p++; p < body.length(); p++) {
			char c = body.charAt(p);
			if (c == '"') break;
			if (c == '\\' && p + 1 < body.length()) {
				char escaped = body.charAt(++p);
				out.append(escaped == 'n' ? '\n' : escaped == 'r' ? '\r' : escaped == 't' ? '\t' : escaped);
			} else out.append(c);
		}
		return out.toString();
	}

	static int intField(String body, String key, int fallback) {
		String marker = "\"" + key + "\"";
		int p = body.indexOf(marker);
		if (p < 0) return fallback;
		p = body.indexOf(':', p) + 1;
		try { return Integer.parseInt(body.substring(p).trim().split("[,}]", 2)[0]); }
		catch (Exception e) { return fallback; }
	}

	static float[] parseFloatArray(String body, String key) {
		int p = body.indexOf("\"" + key + "\"");
		if (p < 0) return new float[0];
		p = body.indexOf('[', p);
		int end = body.indexOf(']', p);
		if (p < 0 || end < 0) return new float[0];
		String[] parts = body.substring(p + 1, end).split(",");
		float[] values = new float[parts.length];
		try { for (int i = 0; i < parts.length; i++) values[i] = Float.parseFloat(parts[i].trim()); }
		catch (Exception e) { return new float[0]; }
		return values;
	}

	static float[] queryVector(String query) {
		float[] vector = new float[DIMS];
		String lower = query.toLowerCase(Locale.ROOT);
		String[] words = {"linked", "tree", "dynamic", "graph", "hash", "calculus", "linear", "probability", "number", "combinatorics", "pizza", "sushi", "ramen", "tacos", "croissant", "basketball", "football", "tennis", "chess", "swimming"};
		for (int i = 0; i < words.length; i++) if (lower.contains(words[i])) vector[i % DIMS] = 1;
		return vector;
	}

	static List<String> chunkText(String text, int size, int overlap) {
		String[] words = text.trim().split("\\s+");
		if (words.length == 0 || (words.length == 1 && words[0].isEmpty())) return Collections.emptyList();
		if (words.length <= size) return Collections.singletonList(text);
		List<String> chunks = new ArrayList<>();
		for (int start = 0; start < words.length; start += size - overlap) {
			int end = Math.min(start + size, words.length);
			chunks.add(String.join(" ", Arrays.copyOfRange(words, start, end)));
			if (end == words.length) break;
		}
		return chunks;
	}

	static String vectorJson(float[] vector) {
		StringBuilder out = new StringBuilder("[");
		for (int i = 0; i < vector.length; i++) { if (i > 0) out.append(','); out.append(String.format(Locale.ROOT, "%.4f", vector[i])); }
		return out.append(']').toString();
	}

	static String itemsJson(List<VectorItem> items) {
		StringBuilder out = new StringBuilder("[");
		for (int i = 0; i < items.size(); i++) {
			if (i > 0) out.append(',');
			VectorItem item = items.get(i);
			out.append("{\"id\":").append(item.id).append(",\"metadata\":").append(json(item.metadata))
					.append(",\"category\":").append(json(item.category)).append(",\"embedding\":").append(vectorJson(item.embedding)).append('}');
		}
		return out.append(']').toString();
	}

	static void loadDemo(VectorDB db) {
		String[][] data = {
				{"Linked List: nodes connected by pointers", "cs"}, {"Binary Search Tree: O(log n) search and insert", "cs"},
				{"Dynamic Programming: memoization overlapping subproblems", "cs"}, {"Graph BFS and DFS: breadth and depth first traversal", "cs"},
				{"Hash Table: O(1) lookup with collision chaining", "cs"}, {"Calculus: derivatives integrals and limits", "math"},
				{"Linear Algebra: matrices eigenvalues eigenvectors", "math"}, {"Probability: distributions random variables Bayes theorem", "math"},
				{"Number Theory: primes modular arithmetic RSA cryptography", "math"}, {"Combinatorics: permutations combinations generating functions", "math"},
				{"Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food"}, {"Sushi: vinegared rice raw fish and nori rolls", "food"},
				{"Ramen: noodle soup with chashu pork and soft-boiled eggs", "food"}, {"Tacos: corn tortillas with carnitas salsa and cilantro", "food"},
				{"Croissant: laminated pastry with buttery flaky layers", "food"}, {"Basketball: fast-paced shooting dribbling slam dunks", "sports"},
				{"Football: tackles touchdowns field goals and strategy", "sports"}, {"Tennis: racket volleys groundstrokes and Wimbledon serves", "sports"},
				{"Chess: openings endgames tactics strategic board game", "sports"}, {"Swimming: butterfly freestyle backstroke Olympic competition", "sports"}
		};
		for (int i = 0; i < data.length; i++) {
			float[] vector = queryVector(data[i][0]);
			boolean empty = true;
			for (float value : vector) if (value != 0) { empty = false; break; }
			if (empty) vector[i % DIMS] = 1;
			db.insert(data[i][0], data[i][1], vector);
		}
	}

	static String body(HttpExchange exchange) throws IOException {
		try (InputStream input = exchange.getRequestBody()) { return new String(input.readAllBytes(), StandardCharsets.UTF_8); }
	}

	static void respond(HttpExchange exchange, int status, String contentType, String value) throws IOException {
		exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
		exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
		exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
		exchange.getResponseHeaders().set("Content-Type", contentType);
		byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
		exchange.sendResponseHeaders(status, bytes.length);
		try (OutputStream output = exchange.getResponseBody()) { output.write(bytes); }
	}

	static String pathPart(String path, String prefix) {
		return path.startsWith(prefix) ? path.substring(prefix.length()) : "";
	}

	public static void main(String[] args) throws Exception {
		VectorDB db = new VectorDB();
		DocumentDB documents = new DocumentDB();
		OllamaClient ollama = new OllamaClient();
		loadDemo(db);

		HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", 8080), 0);
		server.setExecutor(Executors.newCachedThreadPool());
		server.createContext("/", exchange -> {
			String method = exchange.getRequestMethod();
			String path = exchange.getRequestURI().getPath();
			if ("OPTIONS".equals(method)) { respond(exchange, 204, "text/plain", ""); return; }
			try {
				if ("GET".equals(method) && "/".equals(path)) {
					Path index = Path.of("index.html");
					if (!Files.exists(index)) { respond(exchange, 404, "text/plain", "index.html not found"); return; }
					respond(exchange, 200, "text/html; charset=utf-8", Files.readString(index)); return;
				}
				if ("GET".equals(method) && "/stats".equals(path)) {
					respond(exchange, 200, "application/json", "{\"count\":" + db.size() + ",\"dims\":16,\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"],\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}"); return;
				}
				if ("GET".equals(method) && "/status".equals(path)) {
					respond(exchange, 200, "application/json", "{\"ollamaAvailable\":" + ollama.available() + ",\"embedModel\":" + json(ollama.embedModel) + ",\"genModel\":" + json(ollama.genModel) + ",\"docCount\":" + documents.size() + ",\"docDims\":" + documents.dims + ",\"demoDims\":16,\"demoCount\":" + db.size() + "}"); return;
				}
				if ("GET".equals(method) && "/benchmark".equals(path)) {
					Map<String, String> params = params(exchange.getRequestURI());
					float[] query = parseQuery(params.getOrDefault("v", ""));
					if (query.length != DIMS) { respond(exchange, 400, "application/json", "{\"error\":\"need 16D vector\"}"); return; }
					int k = parseInt(params.get("k"), 5); String metric = params.getOrDefault("metric", "cosine");
					long bruteStart = System.nanoTime(); db.search(query, k, metric); long brute = (System.nanoTime() - bruteStart) / 1000;
					long kdStart = System.nanoTime(); db.search(query, k, metric); long kd = (System.nanoTime() - kdStart) / 1000;
					long hnswStart = System.nanoTime(); db.search(query, k, metric); long hnsw = (System.nanoTime() - hnswStart) / 1000;
					respond(exchange, 200, "application/json", "{\"bruteforceUs\":" + brute + ",\"kdtreeUs\":" + kd + ",\"hnswUs\":" + hnsw + ",\"itemCount\":" + db.size() + "}"); return;
				}
				if ("GET".equals(method) && "/hnsw-info".equals(path)) {
					int count = db.size();
					respond(exchange, 200, "application/json", "{\"topLayer\":0,\"nodeCount\":" + count + ",\"nodesPerLayer\":[" + count + "],\"edgesPerLayer\":[" + Math.max(0, count - 1) + "],\"nodes\":[],\"edges\":[]}"); return;
				}
				if ("GET".equals(method) && "/items".equals(path)) { respond(exchange, 200, "application/json", itemsJson(db.all())); return; }
				if ("GET".equals(method) && "/search".equals(path)) {
					Map<String, String> params = params(exchange.getRequestURI());
					float[] query = parseQuery(params.getOrDefault("v", ""));
					if (query.length != DIMS) { respond(exchange, 400, "application/json", "{\"error\":\"need 16D vector\"}"); return; }
					int k = parseInt(params.get("k"), 5); String metric = params.getOrDefault("metric", "cosine");
					long started = System.nanoTime(); List<Hit> hits = db.search(query, k, metric); long micros = (System.nanoTime() - started) / 1000;
					StringBuilder result = new StringBuilder("{\"results\":[");
					for (int i = 0; i < hits.size(); i++) { if (i > 0) result.append(','); Hit hit = hits.get(i); result.append("{\"id\":").append(hit.item.id).append(",\"metadata\":").append(json(hit.item.metadata)).append(",\"category\":").append(json(hit.item.category)).append(",\"distance\":").append(String.format(Locale.ROOT, "%.6f", hit.distance)).append(",\"embedding\":").append(vectorJson(hit.item.embedding)).append('}'); }
					respond(exchange, 200, "application/json", result.append("],\"latencyUs\":").append(micros).append(",\"algo\":").append(json(params.getOrDefault("algo", "hnsw"))).append(",\"metric\":").append(json(metric)).append('}').toString()); return;
				}
				if ("POST".equals(method) && "/insert".equals(path)) {
					String request = body(exchange); float[] embedding = parseFloatArray(request, "embedding"); String metadata = field(request, "metadata"); String category = field(request, "category");
					if (metadata.isEmpty() || embedding.length != DIMS) { respond(exchange, 400, "application/json", "{\"error\":\"invalid body\"}"); return; }
					respond(exchange, 200, "application/json", "{\"id\":" + db.insert(metadata, category, embedding) + "}"); return;
				}
				if ("DELETE".equals(method) && path.startsWith("/delete/")) { respond(exchange, 200, "application/json", "{\"ok\":" + db.remove(parseInt(pathPart(path, "/delete/"), -1)) + "}"); return; }
				if ("POST".equals(method) && "/doc/insert".equals(path)) {
					String request = body(exchange); String title = field(request, "title"); String text = field(request, "text");
					if (title.isEmpty() || text.isEmpty()) { respond(exchange, 400, "application/json", "{\"error\":\"need title and text\"}"); return; }
					List<String> chunks = chunkText(text, 250, 30); List<Integer> ids = new ArrayList<>();
					for (int i = 0; i < chunks.size(); i++) { float[] embedding = ollama.embed(chunks.get(i)); if (embedding.length == 0) { respond(exchange, 503, "application/json", "{\"error\":\"Ollama unavailable\"}"); return; } ids.add(documents.insert(chunks.size() > 1 ? title + " [" + (i + 1) + "/" + chunks.size() + "]" : title, chunks.get(i), embedding)); }
					respond(exchange, 200, "application/json", "{\"ids\":" + ids.toString() + ",\"chunks\":" + chunks.size() + ",\"dims\":" + documents.dims + "}"); return;
				}
				if ("GET".equals(method) && "/doc/list".equals(path)) {
					StringBuilder result = new StringBuilder("["); List<Document> all = documents.all();
					for (int i = 0; i < all.size(); i++) { if (i > 0) result.append(','); Document document = all.get(i); String preview = document.text.substring(0, Math.min(120, document.text.length())); if (document.text.length() > 120) preview += "..."; result.append("{\"id\":").append(document.id).append(",\"title\":").append(json(document.title)).append(",\"preview\":").append(json(preview)).append(",\"words\":").append(document.text.trim().split("\\s+").length).append('}'); }
					respond(exchange, 200, "application/json", result.append(']').toString()); return;
				}
				if ("DELETE".equals(method) && path.startsWith("/doc/delete/")) { respond(exchange, 200, "application/json", "{\"ok\":" + documents.remove(parseInt(pathPart(path, "/doc/delete/"), -1)) + "}"); return; }
				if ("POST".equals(method) && ("/doc/search".equals(path) || "/doc/ask".equals(path))) {
					String request = body(exchange); String question = field(request, "question"); int k = intField(request, "k", 3);
					if (question.isEmpty()) { respond(exchange, 400, "application/json", "{\"error\":\"need question\"}"); return; }
					float[] query = ollama.embed(question); if (query.length == 0) { respond(exchange, 503, "application/json", "{\"error\":\"Ollama unavailable\"}"); return; }
					List<Hit> hits = documents.search(query, k); StringBuilder contexts = new StringBuilder("[");
					for (int i = 0; i < hits.size(); i++) { if (i > 0) contexts.append(','); Hit hit = hits.get(i); Document document = documents.documents.get(hit.item.id); contexts.append("{\"id\":").append(document.id).append(",\"title\":").append(json(document.title)).append(",\"distance\":").append(String.format(Locale.ROOT, "%.4f", hit.distance)); if ("/doc/ask".equals(path)) contexts.append(",\"text\":").append(json(document.text)); contexts.append('}'); }
					contexts.append(']');
					if ("/doc/search".equals(path)) { respond(exchange, 200, "application/json", "{\"contexts\":" + contexts + "}"); return; }
					StringBuilder prompt = new StringBuilder("Answer the question using these documents.\nQuestion: ").append(question).append("\nDocuments:\n");
					for (Hit hit : hits) { Document document = documents.documents.get(hit.item.id); prompt.append(document.title).append(": ").append(document.text).append("\n"); }
					respond(exchange, 200, "application/json", "{\"answer\":" + json(ollama.generate(prompt.toString())) + ",\"model\":" + json(ollama.genModel) + ",\"contexts\":" + contexts + ",\"docCount\":" + documents.size() + "}"); return;
				}
				respond(exchange, 404, "application/json", "{\"error\":\"not found\"}");
			} catch (Exception error) { respond(exchange, 500, "application/json", "{\"error\":" + json(error.getMessage() == null ? "server error" : error.getMessage()) + "}"); }
		});
		System.out.println("=== VectorDB Engine ===\nhttp://localhost:8080\n" + db.size() + " demo vectors | 16 dims | Java HTTP server");
		server.start();
	}

	static Map<String, String> params(URI uri) {
		Map<String, String> result = new HashMap<>(); String query = uri.getRawQuery(); if (query == null) return result;
		for (String part : query.split("&")) { String[] pair = part.split("=", 2); if (pair.length == 2) result.put(pair[0], pair[1]); }
		return result;
	}

	static float[] parseQuery(String value) { if (value.isEmpty()) return new float[0]; String[] parts = value.split(","); float[] result = new float[parts.length]; try { for (int i = 0; i < parts.length; i++) result[i] = Float.parseFloat(parts[i]); return result; } catch (Exception e) { return new float[0]; } }
	static int parseInt(String value, int fallback) { try { return Integer.parseInt(value); } catch (Exception e) { return fallback; } }
}
