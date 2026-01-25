import asyncio
from playwright.async_api import async_playwright

async def scrape_to_pdf(url, output_filename="resultado.pdf"):
    """
    Carga una página web utilizando Playwright, espera a que el contenido
    esté listo y genera un archivo PDF profesional.
    """
    async with async_playwright() as p:
        # Lanzamos el navegador (Chromium es el mejor para exportar a PDF)
        print(f"Iniciando navegador para: {url}...")
        browser = await p.chromium.launch(headless=True)
        
        # Creamos un contexto con un User-Agent moderno para evitar bloqueos
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        page = await context.new_page()

        try:
            # Navegamos a la URL y esperamos a que la red esté inactiva
            # (útil para sitios que cargan mucho JS dinámico)
            await page.goto(url, wait_until="networkidle", timeout=60000)
            
            # Opcional: Desplazamiento automático para activar 'lazy loading' de imágenes
            print("Desplazando para cargar contenido dinámico...")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)  # Pausa breve para renderizado post-scroll

            # Generación del PDF con configuración profesional
            print(f"Generando PDF: {output_filename}...")
            await page.pdf(
                path=output_filename,
                format="A4",
                print_background=True,  # Incluye colores de fondo y sombreados
                margin={"top": "1cm", "right": "1cm", "bottom": "1cm", "left": "1cm"},
                display_header_footer=True,
                footer_template="<div style='font-size:10px; width:100%; text-align:center;'>Página <span class='pageNumber'></span> de <span class='totalPages'></span></div>"
            )
            
            print("¡Proceso completado con éxito!")

        except Exception as e:
            print(f"Error durante el scraping: {e}")
        
        finally:
            await browser.close()

if __name__ == "__main__":
    # Define aquí la URL que deseas procesar
    target_url = "https://www.lamoncloa.gob.es/consejodeministros/referencias/Paginas/2026/20260120-referencia-rueda-de-prensa-ministros.aspx" 
    
    # Ejecutamos el bucle de eventos asíncronos
    asyncio.run(scrape_to_pdf(target_url, "reporte_web.pdf"))