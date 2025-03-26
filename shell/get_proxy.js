const ExtractEnvironmentLink = () => {
  const XPATH = "/html/body/div/div[2]/div[2]/div/div[6]/div/div/div[2]/div/div[8]/div/div/div/div[2]/div/div[5]/div[2]/div/div/div/input";
  const result = document.evaluate(XPATH, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);

  if (result.singleNodeValue) {
    console.log(result.singleNodeValue.value);
  } else {
    console.log("Link Not Found!");
  }
}

ExtractEnvironmentLink();