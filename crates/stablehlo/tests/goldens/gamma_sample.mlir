module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.compare LT, %0, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.add %0, %3 : tensor<f32>
    %6 = stablehlo.select %4, %5, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %8 = stablehlo.subtract %6, %7 : tensor<f32>
    %9 = stablehlo.constant dense<9.0> : tensor<f32>
    %10 = stablehlo.multiply %9, %8 : tensor<f32>
    %11 = stablehlo.sqrt %10 : tensor<f32>
    %12 = stablehlo.divide %3, %11 : tensor<f32>
    %13, %14 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %15 = stablehlo.constant dense<9> : tensor<128xui32>
    %16 = stablehlo.shift_right_logical %14, %15 : tensor<128xui32>
    %17 = stablehlo.convert %16 : (tensor<128xui32>) -> tensor<128xf32>
    %18 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128xf32>
    %20 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %21 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %22 = stablehlo.multiply %19, %20 : tensor<128xf32>
    %23 = stablehlo.subtract %22, %21 : tensor<128xf32>
    %24 = chlo.erf_inv %23 : tensor<128xf32> -> tensor<128xf32>
    %25 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<128xf32>
    %27 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %28 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %29 = stablehlo.multiply %26, %27 : tensor<128xf32>
    %30 = stablehlo.add %29, %28 : tensor<128xf32>
    %31, %32 = stablehlo.rng_bit_generator %13, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %33 = stablehlo.constant dense<9> : tensor<128xui32>
    %34 = stablehlo.shift_right_logical %32, %33 : tensor<128xui32>
    %35 = stablehlo.convert %34 : (tensor<128xui32>) -> tensor<128xf32>
    %36 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %37 = stablehlo.multiply %35, %36 : tensor<128xf32>
    %38 = stablehlo.subtract %3, %2 : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %40 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %41 = stablehlo.multiply %37, %39 : tensor<128xf32>
    %42 = stablehlo.add %41, %40 : tensor<128xf32>
    %43 = stablehlo.constant dense<0> : tensor<i32>
    %44 = stablehlo.constant dense<false> : tensor<i1>
    %45 = stablehlo.constant dense<0.0> : tensor<f32>
    %49:3 = stablehlo.while(%46 = %43, %47 = %44, %48 = %45) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %50 = stablehlo.constant dense<128> : tensor<i32>
      %51 = stablehlo.compare LT, %46, %50, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %52 = stablehlo.not %47 : tensor<i1>
      %53 = stablehlo.and %52, %51 : tensor<i1>
      stablehlo.return %53 : tensor<i1>
    } do {
      %54 = stablehlo.dynamic_slice %30, %46, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %55 = stablehlo.reshape %54 : (tensor<1xf32>) -> tensor<f32>
      %56 = stablehlo.dynamic_slice %42, %46, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %57 = stablehlo.reshape %56 : (tensor<1xf32>) -> tensor<f32>
      %58 = stablehlo.multiply %12, %55 : tensor<f32>
      %59 = stablehlo.add %3, %58 : tensor<f32>
      %60 = stablehlo.multiply %59, %59 : tensor<f32>
      %61 = stablehlo.multiply %60, %59 : tensor<f32>
      %62 = stablehlo.multiply %8, %61 : tensor<f32>
      %63 = stablehlo.constant dense<0.5> : tensor<f32>
      %64 = stablehlo.multiply %55, %55 : tensor<f32>
      %65 = stablehlo.multiply %63, %64 : tensor<f32>
      %66 = stablehlo.multiply %8, %61 : tensor<f32>
      %67 = stablehlo.negate %66 : tensor<f32>
      %68 = stablehlo.log %61 : tensor<f32>
      %69 = stablehlo.multiply %8, %68 : tensor<f32>
      %70 = stablehlo.add %65, %8 : tensor<f32>
      %71 = stablehlo.add %70, %67 : tensor<f32>
      %72 = stablehlo.add %71, %69 : tensor<f32>
      %73 = stablehlo.log %57 : tensor<f32>
      %74 = stablehlo.compare LT, %73, %72 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %75 = stablehlo.compare GT, %61, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %76 = stablehlo.and %74, %75 : tensor<i1>
      %77 = stablehlo.constant dense<1> : tensor<i32>
      %78 = stablehlo.add %46, %77 : tensor<i32>
      stablehlo.return %78, %76, %62 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %79, %80 = stablehlo.rng_bit_generator %31, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %81 = stablehlo.constant dense<9> : tensor<ui32>
    %82 = stablehlo.shift_right_logical %80, %81 : tensor<ui32>
    %83 = stablehlo.convert %82 : (tensor<ui32>) -> tensor<f32>
    %84 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %85 = stablehlo.multiply %83, %84 : tensor<f32>
    %86 = stablehlo.subtract %3, %2 : tensor<f32>
    %87 = stablehlo.multiply %85, %86 : tensor<f32>
    %88 = stablehlo.add %87, %2 : tensor<f32>
    %89 = stablehlo.divide %3, %0 : tensor<f32>
    %90 = stablehlo.power %88, %89 : tensor<f32>
    %91 = stablehlo.select %4, %90, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %92 = stablehlo.multiply %49#2, %91 : tensor<f32>
    %93 = stablehlo.divide %92, %1 : tensor<f32>
    return %93, %79 : tensor<f32>, tensor<2xui64>
  }
}
