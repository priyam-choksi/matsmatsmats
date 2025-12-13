import asyncio
from playwright.async_api import async_playwright
from PIL import Image
import io

async def slides_to_pdf(html_path: str, output_pdf: str, num_slides: int = 13):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': 1920, 'height': 1080})
        
        await page.goto(f'file:///{html_path}')
        await page.wait_for_timeout(2000)  # Let Tailwind load
        
        pdf_pages = []
        
        for slide_num in range(1, num_slides + 1):
            if slide_num > 1:
                await page.keyboard.press('ArrowRight')
                await page.wait_for_timeout(300)
            
            screenshot = await page.screenshot(type='png')
            pdf_pages.append(screenshot)
            print(f'Captured slide {slide_num}/{num_slides}')
        
        await browser.close()
        
        # Combine into PDF
        images = [Image.open(io.BytesIO(png)) for png in pdf_pages]
        images[0].save(
            output_pdf,
            'PDF',
            save_all=True,
            append_images=images[1:],
            resolution=100.0
        )
        print(f'Saved to {output_pdf}')

# Your file
html_file = r'F:\DAMG 7374_GENAI\TradingAgent\diagrams\trade-arena-slides.html'
output_file = r'F:\DAMG 7374_GENAI\TradingAgent\diagrams\trade_arena_slides.pdf'

asyncio.run(slides_to_pdf(html_file, output_file, num_slides=13))