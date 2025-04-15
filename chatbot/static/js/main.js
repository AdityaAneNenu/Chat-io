// Replace the part in your script that handles the display of the response

document.getElementById('send-button').addEventListener('click', async () => {
    const queryInput = document.getElementById('query-input');
    const query = queryInput.value.trim();
    if (!query) return;
  
    document.getElementById('loader').style.display = 'block';
    document.getElementById('response-container').style.display = 'none';
  
    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          mode: currentMode
        }),
      });
  
      const result = await response.json();
      console.log("Response from server:", result);
      
      // Display response
      document.getElementById('response-title').textContent = `Results for: ${query}`;
      document.getElementById('response-source').textContent = `Source: ${result.source || 'Unknown'}`;
      
      // Check if we're getting HTML or markdown
      if (result.format_type === 'html') {
        document.getElementById('response-summary').innerHTML = result.answer || 'No summary available';
      } else {
        // If it's markdown, we need to convert it
        document.getElementById('response-summary').innerHTML = 
          `<div class="formatted-response">${marked.parse(result.answer || 'No summary available')}</div>`;
      }
      
      // Hide the structured data if we're dealing with formatted HTML output
      if (result.format_type === 'html' && !result.structured_data) {
        document.getElementById('response-structured').style.display = 'none';
      } else {
        document.getElementById('response-structured').style.display = 'block';
        document.getElementById('response-structured').textContent = 
          JSON.stringify(result.structured_data || {}, null, 2);
      }
      
      document.getElementById('loader').style.display = 'none';
      document.getElementById('response-container').style.display = 'block';
  
    } catch (err) {
      console.error("Error calling backend:", err);
      alert("Failed to reach server. Please try again later.");
      document.getElementById('loader').style.display = 'none';
    }
  });