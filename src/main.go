package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Node, Edge, and Graph Structs
type Node struct {
	ID        int
	EdgeCount int
	Pattern   int
	Edges     []*Edge
}

type Edge struct {
	ID    int
	Rank  int
	Start *Node
	End   *Node
}

type Graph struct {
	Nodes []Node
	Edges []Edge
}

type SubgraphProfile struct {
	NodeCount       int
	EdgeCount       int
	SortedRanks     []int
	SortedNodeEdges []int
}

// Global variables encapsulated in a struct
type PatternFinder struct {
	parallelProcesses   int64
	startTime           time.Time
	elapsedPatternFinder time.Duration
	elapsedCombination   time.Duration
	subgraphProfile     SubgraphProfile
	wg                  sync.WaitGroup
}

func main() {
	log.Println("Starting the Pattern Finder application")

	pf := &PatternFinder{}
	pf.SetParallelism(4)
	pf.startTime = time.Now()

	graphSize := 10
	subgraphSize := 3
	channelSize := 100
	goroutineCount := 5

	log.Println("Generating main and subgraph")
	mainGraph := GenerateRandomGraph(graphSize)
	subGraph := GenerateRandomGraph(subgraphSize)

	matchChannel := make(chan []int, channelSize)
	outputChannel := make(chan []int, channelSize)

	// Serialize graphs to JSON files
	WriteToFile("Graph.json", ToJSON(mainGraph))
	WriteToFile("SubGraph.json", ToJSON(subGraph))

	edgeIDs := make([]int, len(mainGraph.Edges))
	for i, e := range mainGraph.Edges {
		edgeIDs[i] = e.ID
	}

	log.Println("Starting combination generation goroutine")
	go pf.GenerateCombinations(edgeIDs, subgraphSize-1, matchChannel) // Match subgraph node count to edge combinations

	log.Println("Profiling subgraph for pattern matching")
	pf.ProfileSubgraph(subGraph)

	log.Println("Starting pattern matching goroutines")
	for i := 0; i < goroutineCount; i++ {
		pf.wg.Add(1)
		go pf.FindPatterns(mainGraph, matchChannel, outputChannel)
	}

	// Collect and write output after pattern matching is done
	go func() {
		pf.wg.Wait()
		close(outputChannel)
		log.Println("All pattern matching goroutines have finished")
	}()

	// Write all found patterns to the file
	pf.WritePatternOutput(*mainGraph, outputChannel)

	var input string
	fmt.Scanln(&input)
	log.Println("Pattern Finder Duration: ", pf.elapsedPatternFinder)
	log.Println("Exiting the application")
}

// SetParallelism configures the number of CPU cores
func (pf *PatternFinder) SetParallelism(cores int) {
	totalCores := runtime.NumCPU()
	log.Printf("Available Cores: %d", totalCores)

	if cores > totalCores {
		cores = totalCores
	}
	runtime.GOMAXPROCS(cores)
	log.Printf("Using %d cores for parallelism", cores)
}

// ProfileSubgraph creates a profile of the subgraph to be used in matching
func (pf *PatternFinder) ProfileSubgraph(graph *Graph) {
	log.Println("Profiling the subgraph")
	pf.subgraphProfile.NodeCount = len(graph.Nodes)
	pf.subgraphProfile.EdgeCount = len(graph.Edges)

	edgeRanks := make([]int, len(graph.Edges))
	for i, e := range graph.Edges {
		edgeRanks[i] = e.Rank
	}
	sort.Sort(sort.Reverse(sort.IntSlice(edgeRanks)))
	pf.subgraphProfile.SortedRanks = edgeRanks

	nodeEdges := make([]int, len(graph.Nodes))
	for i, n := range graph.Nodes {
		nodeEdges[i] = n.EdgeCount
	}
	sort.Sort(sort.Reverse(sort.IntSlice(nodeEdges)))
	pf.subgraphProfile.SortedNodeEdges = nodeEdges

	log.Printf("Subgraph profile: %d nodes, %d edges", pf.subgraphProfile.NodeCount, pf.subgraphProfile.EdgeCount)
}

// GenerateRandomGraph creates a random graph of a given size
func GenerateRandomGraph(size int) *Graph {
	log.Printf("Generating random graph of size %d", size)
	graph := new(Graph)
	graph.Nodes = make([]Node, size)
	graph.Edges = make([]Edge, 0)

	if size < 2 {
		log.Println("Graph size is too small, returning empty graph")
		return graph
	}

	var edgeID int

	for i := 0; i < size; i++ {
		graph.Nodes[i].ID = i

		randomIndex := rand.Intn(size)
		for randomIndex == i {
			randomIndex = rand.Intn(size)
		}

		edge := Edge{ID: edgeID, Start: &graph.Nodes[i], End: &graph.Nodes[randomIndex]}
		edgeID++

		graph.Edges = append(graph.Edges, edge)
		graph.Nodes[i].Edges = append(graph.Nodes[i].Edges, &edge)
		graph.Nodes[i].EdgeCount++
		graph.Nodes[randomIndex].Edges = append(graph.Nodes[randomIndex].Edges, &edge)
		graph.Nodes[randomIndex].EdgeCount++
	}

	for i := range graph.Edges {
		edge := &graph.Edges[i]
		edge.Rank = edge.Start.EdgeCount + edge.End.EdgeCount
	}

	log.Printf("Generated graph with %d nodes and %d edges", len(graph.Nodes), len(graph.Edges))
	return graph
}

// GenerateCombinations generates all combinations of r edges
func (pf *PatternFinder) GenerateCombinations(iterable []int, r int, ch chan []int) {
	log.Println("Starting combination generation")
	pool := iterable
	n := len(pool)

	if r > n {
		log.Println("Combination size is larger than the pool, returning")
		close(ch)
		return
	}

	indices := make([]int, r)
	for i := range indices {
		indices[i] = i
	}

	for {
		result := make([]int, r)
		for i, el := range indices {
			result[i] = pool[el]
		}
		ch <- result

		// Find the rightmost index that can be incremented
		i := r - 1
		for i >= 0 && indices[i] == i+n-r {
			i--
		}

		if i < 0 {
			pf.elapsedCombination = time.Since(pf.startTime)
			log.Println("Finished generating combinations")
			close(ch)
			return
		}

		// Increment this index and reset the following ones
		indices[i]++
		for j := i + 1; j < r; j++ {
			indices[j] = indices[j-1] + 1
		}
	}
}

// FindPatterns checks for subgraph pattern matches and sends matches to the output channel
func (pf *PatternFinder) FindPatterns(graph *Graph, combinationCh, outputCh chan []int) {
	defer pf.wg.Done()
	defer atomic.AddInt64(&pf.parallelProcesses, 1)

	log.Println("Started pattern matching goroutine")
	for combination := range combinationCh {
		log.Printf("Checking combination: %v", combination)
		if pf.isPatternMatch(graph, combination) {
			log.Printf("Found match: %v", combination)
			outputCh <- combination
		} else {
			log.Printf("No match found for combination: %v", combination)
		}
	}

	if atomic.LoadInt64(&pf.parallelProcesses) == 0 {
		pf.elapsedPatternFinder = time.Since(pf.startTime)
		log.Println("All pattern matching processes are done")
	}
}

// isPatternMatch checks if a combination of edges matches the subgraph
func (pf *PatternFinder) isPatternMatch(graph *Graph, combination []int) bool {
	edges := pf.extractEdges(graph, combination)
	allNodes := pf.getAllNodesFromEdges(edges)

	if len(allNodes) != pf.subgraphProfile.NodeCount {
		log.Printf("Node count mismatch: expected %d, got %d", pf.subgraphProfile.NodeCount, len(allNodes))
		return false
	}

	if !pf.isValidEdgeRank(edges) || !pf.isValidNodeEdgeCount(allNodes) {
		log.Println("Edge rank or node edge count mismatch")
		return false
	}

	return true
}

// Extracts edges from the graph based on a combination of edge IDs
func (pf *PatternFinder) extractEdges(graph *Graph, combination []int) []Edge {
	edges := make([]Edge, len(combination))
	for i, edgeID := range combination {
		edges[i] = graph.Edges[edgeID]
	}
	return edges
}

// getAllNodesFromEdges retrieves all unique nodes involved in a set of edges
func (pf *PatternFinder) getAllNodesFromEdges(edges []Edge) []Node {
	nodeSet := make(map[int]*Node)  // Use map to store unique nodes
	for _, edge := range edges {
		// Insert both the start and end nodes into the map
		nodeSet[edge.Start.ID] = edge.Start
		nodeSet[edge.End.ID] = edge.End
	}
    
	// Convert map values to a slice of unique nodes
	allNodes := make([]Node, 0, len(nodeSet))
	for _, node := range nodeSet {
		allNodes = append(allNodes, *node)
	}

	log.Printf("Extracted %d unique nodes from edges", len(allNodes))
	return allNodes
}

// isValidEdgeRank checks if the edge ranks match the subgraph profile
func (pf *PatternFinder) isValidEdgeRank(edges []Edge) bool {
	edgeRanks := make([]int, len(edges))
	for i, edge := range edges {
		edgeRanks[i] = edge.Rank
	}
	sort.Sort(sort.Reverse(sort.IntSlice(edgeRanks)))

	for i, rank := range edgeRanks {
		if rank < pf.subgraphProfile.SortedRanks[i] {
			log.Printf("Edge rank mismatch: expected >= %d, got %d", pf.subgraphProfile.SortedRanks[i], rank)
			return false
		}
	}
	return true
}

// isValidNodeEdgeCount checks if the node edge counts match the subgraph profile
func (pf *PatternFinder) isValidNodeEdgeCount(nodes []Node) bool {
	nodeEdgeCounts := make([]int, len(nodes))
	for i, node := range nodes {
		nodeEdgeCounts[i] = node.EdgeCount
	}
	sort.Sort(sort.Reverse(sort.IntSlice(nodeEdgeCounts)))

	for i, edgeCount := range nodeEdgeCounts {
		if edgeCount < pf.subgraphProfile.SortedNodeEdges[i] {
			log.Printf("Node edge count mismatch: expected >= %d, got %d", pf.subgraphProfile.SortedNodeEdges[i], edgeCount)
			return false
		}
	}
	return true
}

// WritePatternOutput writes the matched patterns to a file
func (pf *PatternFinder) WritePatternOutput(graph Graph, outputCh chan []int) {
	log.Println("Starting to write pattern output")
	matches := [][]int{}
	for combination := range outputCh {
		matches = append(matches, combination)
		log.Printf("Writing match: %v", combination)
	}

	if len(matches) > 0 {
		WriteToFile("PatternGraph.json", ToJSON(&graph))
		log.Println("Pattern output written to file")
	} else {
		log.Println("No matches found")
	}
}

// WriteToFile writes content to a specified file
func WriteToFile(filename string, content string) {
	log.Printf("Writing to file: %s", filename)
	err := ioutil.WriteFile(filename, []byte(content), 0644)
	if err != nil {
		log.Fatalf("Failed to write to file: %s, error: %v", filename, err)
	}
}

// ToJSON converts the graph into a JSON representation
func ToJSON(graph *Graph) string {
	type NodeJSON struct {
		Id    int `json:"id"`
		Group int `json:"group"`
	}

	type EdgeJSON struct {
		Source int `json:"source"`
		Target int `json:"target"`
		Value  int `json:"value"`
	}

	nodes := make([]NodeJSON, len(graph.Nodes))
	for i, node := range graph.Nodes {
		nodes[i] = NodeJSON{Id: node.ID, Group: node.Pattern}
	}

	edges := make([]EdgeJSON, len(graph.Edges))
	for i, edge := range graph.Edges {
		edges[i] = EdgeJSON{Source: edge.Start.ID, Target: edge.End.ID, Value: int(math.Ceil(float64(edge.Rank) / 2.0))}
	}

	nodesJSON, _ := json.Marshal(nodes)
	edgesJSON, _ := json.Marshal(edges)

	log.Println("Graph serialized to JSON")
	return fmt.Sprintf("{ \"nodes\": %s, \"links\": %s }", nodesJSON, edgesJSON)
}
