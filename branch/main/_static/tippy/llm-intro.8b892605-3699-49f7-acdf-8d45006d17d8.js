selector_to_html = {"a[href=\"https://doi.org/10.1145/3442188.3445922\"]": "\n<div>\n    <h3>On the Dangers of Stochastic Parrots</h3>\n    \n    <p><b>Authors:</b> Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell</p>\n    \n    <p><b>Publisher:</b> ACM</p>\n    <p><b>Published:</b> 2021-3-2</p>\n</div>", "a[href=\"#id1\"]": "<figure class=\"align-default\" id=\"id1\">\n<a class=\"reference internal image-reference\" href=\"../_images/llm.png\"><img alt=\"Input tokens -&gt;  Model -&gt; Input tokens + output token\" src=\"../_images/llm.png\" style=\"width: 100%;\"/>\n</a>\n<figcaption>\n<p><span class=\"caption-text\">A simplistic mental model for an LLM</span><a class=\"headerlink\" href=\"#id1\" title=\"Link to this image\">\u00b6</a></p>\n</figcaption>\n</figure>", "a[href=\"#what-is-a-large-language-model\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">What is a Large Language Model?<a class=\"headerlink\" href=\"#what-is-a-large-language-model\" title=\"Link to this heading\">\u00b6</a></h1><p>A Large Language Model (LLM) is a type of artificial intelligence trained on\nvast amounts of text data to predict and generate human-like text. At their\ncore, these models learn statistical patterns in language: given a sequence of\nwords (or better <em>tokens</em>: fragments of words, commas, and anything in text),\nthey predict what comes next.</p>"}
skip_classes = ["headerlink", "sd-stretched-link"]

window.onload = function () {
    for (const [select, tip_html] of Object.entries(selector_to_html)) {
        const links = document.querySelectorAll(` ${select}`);
        for (const link of links) {
            if (skip_classes.some(c => link.classList.contains(c))) {
                continue;
            }

            tippy(link, {
                content: tip_html,
                allowHTML: true,
                arrow: true,
                placement: 'auto-start', maxWidth: 500, interactive: false,

            });
        };
    };
    console.log("tippy tips loaded!");
};
