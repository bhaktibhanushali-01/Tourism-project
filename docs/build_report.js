const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, ImageRun, AlignmentType, PageBreak, LevelFormat,
  Header, Footer, PageNumber, TabStopType, TabStopPosition
} = require("docx");

const FIG = (name) => path.join(__dirname, "figures", name);
const img = (name, w, h) => new ImageRun({ type: "png", data: fs.readFileSync(FIG(name)), transformation: { width: w, height: h } });

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 250, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 120 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 18, color: "555555" })],
    alignment: AlignmentType.CENTER, spacing: { after: 300 },
  });
}
function centerImage(name, w, h) {
  return new Paragraph({ children: [img(name, w, h)], alignment: AlignmentType.CENTER, spacing: { before: 150, after: 60 } });
}

function simpleTable(headers, rows) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map(hd => new TableCell({
      width: { size: 100 / headers.length, type: WidthType.PERCENTAGE },
      shading: { type: ShadingType.CLEAR, fill: "2F5496" },
      children: [new Paragraph({ children: [new TextRun({ text: hd, bold: true, color: "FFFFFF", size: 20 })] })],
    })),
  });
  const dataRows = rows.map((r, i) => new TableRow({
    children: r.map(cell => new TableCell({
      width: { size: 100 / headers.length, type: WidthType.PERCENTAGE },
      shading: { type: ShadingType.CLEAR, fill: i % 2 === 0 ? "F2F2F2" : "FFFFFF" },
      children: [new Paragraph({ children: [new TextRun({ text: String(cell), size: 20 })] })],
    })),
  }));
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [headerRow, ...dataRows],
  });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
  },
  sections: [
    // ---------------- Title Page ----------------
    {
      properties: {},
      children: [
        new Paragraph({ text: "", spacing: { before: 2000 } }),
        new Paragraph({
          children: [new TextRun({ text: "Tourism Experience Analytics", bold: true, size: 56, color: "1F4E79" })],
          alignment: AlignmentType.CENTER, spacing: { after: 200 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Classification, Prediction, and Recommendation System", size: 32, color: "2F5496" })],
          alignment: AlignmentType.CENTER, spacing: { after: 800 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Project Report", size: 28, bold: true })],
          alignment: AlignmentType.CENTER, spacing: { after: 100 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Domain: Tourism", size: 24, italics: true, color: "555555" })],
          alignment: AlignmentType.CENTER, spacing: { after: 2000 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Prepared using: Python · pandas · scikit-learn · LightGBM · Streamlit", size: 20, color: "777777" })],
          alignment: AlignmentType.CENTER,
        }),
        new Paragraph({ children: [new PageBreak()] }),

        // ---------------- Table of Contents (manual) ----------------
        h1("Table of Contents"),
        p("1. Executive Summary"),
        p("2. Problem Statement & Business Use Cases"),
        p("3. Data Overview & Sources"),
        p("4. Data Cleaning & Preprocessing"),
        p("5. Exploratory Data Analysis"),
        p("6. Feature Engineering"),
        p("7. Model Building & Evaluation"),
        p("8. Recommendation System"),
        p("9. Streamlit Application"),
        p("10. Key Business Insights"),
        p("11. Known Limitations & Next Steps"),
        p("12. Conclusion"),
        new Paragraph({ children: [new PageBreak()] }),

        // ---------------- 1. Executive Summary ----------------
        h1("1. Executive Summary"),
        p("This project analyzes 52,930 tourist visit transactions covering 33,530 unique users and 30 attractions across Bali and Java, Indonesia (2013–2022), to build an end-to-end analytics system for a tourism platform. Three machine-learning capabilities were developed:"),
        bullet("A regression model to predict the rating a visit is likely to receive."),
        bullet("A classification model to predict the mode of a visit (Business, Couples, Family, Friends, Solo)."),
        bullet("A recommendation engine (collaborative + content-based filtering) to suggest attractions to users."),
        p("All three capabilities are deployed together in an interactive Streamlit web application, alongside a full exploratory data analysis dashboard."),

        // ---------------- 2. Problem Statement ----------------
        h1("2. Problem Statement & Business Use Cases"),
        p("Tourism agencies and travel platforms aim to enhance user experience by leveraging data to provide personalized recommendations, predict user satisfaction, and understand visitor behavior. This project addresses four business use cases:"),
        bullet("Personalized Recommendations — suggest attractions aligned with a user's preferences and visit history."),
        bullet("Tourism Analytics — surface insights into popular attractions, cities and countries for planning."),
        bullet("Customer Segmentation — understand visit-mode patterns (who travels how, and when) for targeted promotions."),
        bullet("Customer Retention — improve loyalty by anticipating what a user is likely to enjoy."),

        // ---------------- 3. Data Overview ----------------
        h1("3. Data Overview & Sources"),
        p("Eight source tables were supplied and merged into a single analytical dataset:"),
        simpleTable(
          ["Table", "Rows", "Purpose"],
          [
            ["Transaction", "52,930", "Core fact table: user, attraction, visit date, mode, rating"],
            ["Item", "30", "Attraction name, address, type, city"],
            ["Type", "17", "Attraction category lookup (Beaches, Museums, etc.)"],
            ["Mode", "6", "Visit mode lookup (Business/Couples/Family/Friends/Solo)"],
            ["City", "9,143", "City name & country"],
            ["Country", "165", "Country name & region"],
            ["Region", "22", "Region name & continent"],
            ["Continent", "6", "Continent name"],
          ]
        ),
        new Paragraph({ text: "", spacing: { after: 150 } }),
        p("Note on scope: the project brief also describes a User table containing each user's own Continent/Region/Country/City. That table was not included in the files provided for this build — see Section 11 for the impact and remediation plan.", { italics: true, color: "B00020" }),
        p("Data quality finding: Item.xlsx's AttractionCityId (values 1, 2, 3) does not reference the same id space as City.xlsx's CityId. A naive join (as the raw schema implies) silently resolves every Bali attraction to \u201cDouala, Cameroon\u201d, every Malang attraction to \u201cSouth Region, Cameroon\u201d, and every Yogyakarta attraction to \u201cN'Djamena, Chad\u201d. This was caught by cross-checking against the AttractionAddress text field and corrected \u2014 see Section 4.3.", { italics: true, color: "B00020" }),

        // ---------------- 4. Data Cleaning ----------------
        h1("4. Data Cleaning & Preprocessing"),
        h2("4.1 Cleaning steps performed"),
        bullet("Removed exact duplicate transactions (none found — data was already de-duplicated on TransactionId)."),
        bullet("Validated Rating is within 1–5, VisitMonth within 1–12, VisitYear within a sane range; no invalid rows found."),
        bullet("Replaced '-' placeholder text (used for unknown/root records with id 0) with an explicit 'Unknown' label across City, Country, Region and Continent tables, and filled the one missing CityName."),
        bullet("Joined all 8 tables into a single master dataset (52,930 rows × 22 columns) keyed on AttractionId, AttractionCityId, AttractionCountryId, AttractionRegionId and AttractionContinentId."),
        bullet("Derived VisitDate (from year+month) and VisitSeason (Winter/Spring/Summer/Autumn) for time-based analysis."),
        h2("4.2 Result"),
        p("Zero missing values remained in the final merged dataset across all 22 columns."),

        // ---------------- 5. EDA ----------------
        h1("5. Exploratory Data Analysis"),
        h2("5.1 Ratings"),
        centerImage("01_rating_distribution.png", 420, 280),
        caption("Figure 1 — Rating distribution: ratings skew positive (avg 4.16/5); over 78% of visits receive a 4 or 5."),

        h2("5.2 Visit Modes"),
        centerImage("02_visit_mode_distribution.png", 420, 280),
        caption("Figure 2 — Couples (40.8%) and Family (28.7%) dominate visit modes; Business trips are rare (1.2%)."),

        h2("5.3 Top Attractions"),
        centerImage("03_top_attractions.png", 420, 300),
        caption("Figure 3 — Sacred Monkey Forest Sanctuary is the single most-visited attraction (13,198 visits), followed by Waterbom Bali and Tegalalang Rice Terrace."),

        h2("5.4 Rating by Attraction Type"),
        centerImage("04_avg_rating_by_type.png", 420, 360),
        caption("Figure 4 — Water Parks (4.65), Spas (4.57) and Caverns & Caves (4.50) earn the highest average ratings."),

        new Paragraph({ children: [new PageBreak()] }),
        h2("5.5 Visits Over Time"),
        centerImage("05_visits_by_year.png", 420, 260),
        caption("Figure 5 — Visit volume by year (2013–2022)."),

        h2("5.6 Seasonality"),
        centerImage("06_visits_by_season.png", 420, 280),
        caption("Figure 6 — Visits are fairly balanced across seasons, with a mild Summer peak (26.9%)."),

        h2("5.7 Rating by Visit Mode"),
        centerImage("07_rating_by_mode.png", 420, 280),
        caption("Figure 7 — Business travelers give the highest average ratings (4.31); Solo travelers the lowest (4.09)."),

        h2("5.8 Top Cities"),
        centerImage("08_top_cities.png", 420, 300),
        caption("Figure 8 — Visit volume concentrated in three cities across Bali/Java."),

        h2("5.9 Correlation Heatmap"),
        centerImage("09_correlation_heatmap.png", 320, 260),
        caption("Figure 9 — Numeric features (year, month, visit mode) show weak linear correlation with rating, motivating the use of tree-based, non-linear models."),

        // ---------------- 6. Feature Engineering ----------------
        h1("6. Feature Engineering"),
        bullet("AttractionPopularity — number of historical visits per attraction (proxy for demand/trust)."),
        bullet("AttractionAvgRatingHist — historical average rating per attraction."),
        bullet("VisitSeason — derived from VisitMonth for seasonality analysis."),
        bullet("Categorical encoding: AttractionTypeId, AttractionCityId, AttractionCountryId, VisitMode used as native integer codes for tree-based models (which handle this natively without one-hot inflation)."),

        // ---------------- 7. Model Building ----------------
        h1("7. Model Building & Evaluation"),
        h2("7.1 Regression — Predicting Rating"),
        p("Three algorithms were trained and compared on an 80/20 train-test split:"),
        simpleTable(
          ["Model", "R²", "RMSE", "MAE"],
          [
            ["Linear Regression", "0.041", "0.950", "0.755"],
            ["Random Forest", "0.089", "0.926", "0.725"],
            ["Gradient Boosting", "0.117", "0.912", "0.716"],
          ]
        ),
        new Paragraph({ text: "", spacing: { after: 150 } }),
        p("Gradient Boosting was selected as the production model. Accuracy is modest — this is expected given the absence of user demographic features (see Section 11); the model still adds value by capturing attraction-type and popularity effects."),

        h2("7.2 Classification — Predicting Visit Mode"),
        simpleTable(
          ["Model", "Accuracy", "Precision", "Recall", "F1-score"],
          [
            ["Logistic Regression", "0.419", "0.367", "0.419", "0.307"],
            ["Random Forest", "0.332", "0.403", "0.332", "0.358"],
            ["LightGBM", "0.468", "0.452", "0.468", "0.411"],
          ]
        ),
        new Paragraph({ text: "", spacing: { after: 150 } }),
        p("LightGBM was selected as the production model (46.8% accuracy vs. a 40.8% naive baseline of always predicting the majority class 'Couples'). Couples and Family are predicted more reliably than Business (rare class, only 1.2% of data)."),

        // ---------------- 8. Recommendation System ----------------
        h1("8. Recommendation System"),
        h2("8.1 Collaborative Filtering"),
        p("Item-based collaborative filtering computes cosine similarity between attractions based on the 33,530 × 30 user-item rating matrix. For a given user, attractions similar to what they've already rated highly are recommended."),
        h2("8.2 Content-Based Filtering"),
        p("Cosine similarity over one-hot encoded AttractionType and AttractionCity — enables recommending attractions to brand-new users given just one attraction they've liked (solves the cold-start problem for collaborative filtering)."),
        h2("8.3 Popularity Fallback"),
        p("A weighted popularity score (average rating × log of visit count) is used as a default recommendation list for entirely new users with no interaction history at all."),

        // ---------------- 9. Streamlit App ----------------
        h1("9. Streamlit Application"),
        p("A six-page interactive web application was built to serve all models:"),
        simpleTable(
          ["Page", "Function"],
          [
            ["Overview", "Headline KPIs and summary charts"],
            ["Explore the Data", "Interactive EDA across attractions, geography, time, ratings"],
            ["Predict Rating", "Regression model — live rating prediction"],
            ["Predict Visit Mode", "Classification model — live visit-mode prediction with probability chart"],
            ["Get Recommendations", "Collaborative, content-based, and popularity-based recommendations"],
            ["About", "Project documentation and known limitations"],
          ]
        ),
        new Paragraph({ text: "", spacing: { after: 150 } }),
        p("Run with: streamlit run app/app.py", { font: "Consolas", size: 20 }),

        // ---------------- 10. Key Insights ----------------
        h1("10. Key Business Insights"),
        bullet("Sacred Monkey Forest Sanctuary is the flagship attraction, driving 25% of all visits — prioritize its capacity planning and satisfaction monitoring."),
        bullet("Couples (40.8%) and Family (28.7%) are the two dominant visit segments — marketing and packages should center on these groups."),
        bullet("Water Parks, Spas and Caverns & Caves earn the highest satisfaction (4.5+ average) despite lower visit volume — these are under-promoted high-satisfaction categories worth featuring more prominently."),
        bullet("Solo travelers report the lowest average ratings (4.09) — investigate whether solo-traveler experience (queues, pricing, safety) needs improvement."),
        bullet("Visit volume is stable across seasons — no major seasonal marketing gap, though Summer holds a slight edge."),

        // ---------------- 11. Limitations ----------------
        h1("11. Known Limitations & Next Steps"),
        p("Missing User demographic table: the project brief calls for a User table linking each user to their home Continent/Region/Country/City. This table was not among the files provided, so:", { bold: true }),
        bullet("Regression and classification models rely only on visit-context and attraction features — no personalization by user origin."),
        bullet("This is the primary driver of the moderate R² (0.117) and accuracy (46.8%) scores."),
        p("Recommendation: source or reconstruct the User table, add user demographic features to both models, and retrain — expected to materially improve both regression and classification performance."),
        p("Other next steps:"),
        bullet("Expand the attraction catalog beyond the current 30 Bali/Java attractions for broader coverage."),
        bullet("Experiment with matrix-factorization (SVD/ALS) collaborative filtering for larger-scale deployments."),
        bullet("Add A/B testing instrumentation once the recommendation system is live."),

        // ---------------- 12. Conclusion ----------------
        h1("12. Conclusion"),
        p("This project delivers a complete, production-shaped analytics pipeline — from raw multi-table data through cleaning, EDA, three trained ML models, and a live Streamlit application — fulfilling all core deliverables from the project brief (cleaned dataset, source code, application, documentation) within the given scope and data constraints."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(__dirname, "Project_Report.docx"), buf);
  console.log("Report written.");
});
