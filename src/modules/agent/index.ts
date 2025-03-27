import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'
import initAgent from './agent'
import { initGraph } from '../graph'

// tag::call[]
export async function call(input: string, sessionId: string): Promise<string> {
    const llm = new ChatOpenAI({
        openAIApiKey: process.env.OPENAI_API_KEY,
        configuration: {
            baseURL: process.env.OPENAI_API_BASE,
        },
    })
    const embeddings = new OpenAIEmbeddings({
        openAIApiKey: process.env.OPENAI_API_KEY,
        configuration: {
            baseURL: process.env.OPENAI_API_BASE,
        },
    })

    // Get Graph Singleton
    const graph = await initGraph()

    const agent = await initAgent(llm, embeddings, graph)
    return await agent.invoke({ input }, { configurable: { sessionId } })
}

// end::call[]
